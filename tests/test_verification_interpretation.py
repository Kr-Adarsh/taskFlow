"""Semantic repair stays bounded and cannot stand in for independent checks."""
from copy import deepcopy
import json

import pytest

from backend.app.agent.provider import FakeProvider, ProviderError
from backend.app.agent.schemas import VerificationIntent, SummaryAssessment
from backend.app.agent.verifier import snapshot_state, VerifierEngine
from backend.app.agent.interpretation import interpret_contract, ContractInterpretationError
from backend.app.agent_v2.context import ContextMemory, decision_prompt
from backend.app.agent_v2.state import Subtask, GraphState
from backend.app.agent_v2.verifier import CapabilityVerifier
from backend.app.capabilities.python import PythonCapability
from backend.app.capabilities.python.verification import CalculationContract
from backend.app.workspace.db import get_db_connection
from backend.app.workspace.models import SupportTicketCreate
from backend.app.workspace.seed import seed_workspace
from backend.app.workspace.service import create_support_ticket

COMPLAINT = 'Read complaint 4821, identify the customer, check their CRM account, and if they are an Enterprise customer, create a high-priority support ticket summarizing the complaint.'
DATASET = 'Analyze sales.csv and identify the region with the largest absolute revenue decline between 2026-08 and 2026-09, including the calculated decline.'
# The missing fields and diagnostic claims match the preserved manual run.
BAD_TICKET = VerificationIntent(collection='tickets', unsupported_criteria=['Read complaint 4821', 'summarizing the complaint'])
GOOD_TICKET = VerificationIntent(collection='tickets', complaint_id='4821', priority='High', condition_tier='Enterprise')
CALCULATION = dict(document_id='sales.csv', group_column='region', value_column='revenue', period_column='month',
                   baseline_period='2026-08', current_period='2026-09', measure='difference',
                   convention='baseline_minus_current', selection='max', group_metric='region', value_metric='decline')


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    path = tmp_path / 'contracts.db'
    monkeypatch.setenv('TASKFLOW_DB_PATH', str(path))
    monkeypatch.setenv('TASKFLOW_ARTIFACTS_DIR', str(tmp_path / 'artifacts'))
    seed_workspace(path, force_reseed=True)
    return path


def assessment():
    return SummaryAssessment(accurate=True, reason='The source describes the outage',
                            source_quotes=['severe database synchronization outages'], contradictions=[])


@pytest.mark.anyio
async def test_failed_ticket_interpretation_is_repaired_before_independent_checks(workspace):
    before = snapshot_state(workspace)
    create_support_ticket(SupportTicketCreate(customer='Acme Corp', priority='High', source_reference='complaint_4821.txt',
                              summary='Severe database synchronization outages across EU servers.'), workspace)
    provider = FakeProvider([deepcopy(BAD_TICKET), deepcopy(GOOD_TICKET), assessment()])
    verifier = CapabilityVerifier(provider, workspace)
    task = Subtask(task_id='ticket', goal=COMPLAINT, success_criteria=[COMPLAINT], verification_capability='browser')
    result = await verifier.verify(COMPLAINT, task, before, ContextMemory())
    assert result.verified and not provider.responses
    evidence = result.evidence['verification_result']
    assert all(item['passed'] for item in evidence['criteria_results'])
    attempts = evidence['context']['interpretation_attempts']
    assert [item['admitted'] for item in attempts] == [False, True]
    assert attempts[0]['contract'] == BAD_TICKET.model_dump()
    repair = json.loads(provider.call_history[1][-1]['content'])['interpretation_repair']
    assert repair['previous_contract']['complaint_id'] is None
    assert all(value in repair['admission_error'] for value in ['4821', 'High', 'Enterprise'])
    assert json.loads(provider.call_history[1][1]['content'])['original_objective'] == COMPLAINT
    assert 'TIK-' not in str(provider.call_history[:2])
    assert verifier.browser.intent == GOOD_TICKET
    task.result['verification'] = result.model_dump()
    assert verifier.audit_mutations([task], before, snapshot_state(workspace)).verified


@pytest.mark.anyio
@pytest.mark.parametrize('changes', [{'complaint_id': '4822'}, {'priority': None}, {'priority': 'Low'}, {'condition_tier': None}, {'condition_tier': 'Standard'}])
async def test_repair_cannot_drop_explicit_source_priority_or_condition(workspace, changes):
    bad = GOOD_TICKET.model_copy(update=changes)
    provider = FakeProvider([bad, deepcopy(bad)])
    verifier = CapabilityVerifier(provider, workspace)
    task = Subtask(task_id='ticket', goal=COMPLAINT, success_criteria=[COMPLAINT], verification_capability='browser')
    result = await verifier.verify(COMPLAINT, task, snapshot_state(workspace), ContextMemory())
    assert result.outcome == 'FATAL_FAILURE' and not result.verified
    assert len(provider.call_history) == 2 and not provider.responses
    assert verifier.browser.intent is None
    failure = result.evidence['verification_result']['context']['interpretation_failure']
    assert failure['origin'] == 'verifier_interpretation'


@pytest.mark.anyio
async def test_corrected_contract_does_not_pass_missing_ticket(workspace):
    provider = FakeProvider([deepcopy(BAD_TICKET), deepcopy(GOOD_TICKET)])
    verifier = VerifierEngine(workspace, provider)
    result = await verifier.verify_run(COMPLAINT, {})
    assert not result.verified and result.context['interpretation_failure'] is None
    assert any('No support ticket' in item for item in result.discrepancies)


@pytest.mark.anyio
async def test_provider_failure_is_not_relabelled_as_interpretation_ambiguity():
    class Unavailable:
        async def generate_structured(self, *args):
            raise ProviderError('Transport unavailable')
    with pytest.raises(ProviderError):
        await interpret_contract(Unavailable(), [], VerificationIntent, lambda value: None)


def replace_sales(workspace, tmp_path):
    path = tmp_path / 'sales.csv'
    # Growth is greater than every decline; absolute movement would pick Growth.
    path.write_text('region,month,revenue\nGrowth,2026-08,100\nGrowth,2026-09,1100\nLoss,2026-08,500\nLoss,2026-09,300\n')
    with get_db_connection(workspace) as connection:
        connection.execute("UPDATE documents_index SET filepath=? WHERE filename='sales.csv'", (str(path),))
        connection.commit()


async def computation(workspace, tmp_path, provider, code):
    replace_sales(workspace, tmp_path)
    memory = ContextMemory()
    capability = PythonCapability(run_id='contract-test')
    profile = capability.profile_dataset('sales.csv')
    assert profile.ok
    memory.update('profile_dataset', {'document_id': 'sales.csv'}, profile)
    result = await capability.execute_python(['sales.csv'], code)
    assert result.ok, result
    task = Subtask(task_id='data', goal=DATASET, success_criteria=[DATASET], verification_capability='python',
                   result={key: result.data[key] for key in ['summary', 'metrics', 'tables']})
    before = snapshot_state(workspace)
    verified = await CapabilityVerifier(provider, workspace).verify(DATASET, task, before, memory, result.data)
    return verified, result


def program(absolute=False):
    return """df=pd.read_csv(inputs['sales.csv'])
totals=df.groupby(['region','month'])['revenue'].sum().unstack()
decline=totals['2026-08']-totals['2026-09']
""" + ('decline=decline.abs()\n' if absolute else '') + "result={'summary':'Compared period totals','metrics':{'region':str(decline.idxmax()),'decline':float(decline.max())},'tables':{'comparison':totals.reset_index()}}"


@pytest.mark.anyio
async def test_spurious_unsupported_decline_gets_one_repair_and_full_recomputation(workspace, tmp_path):
    claim = 'absolute value across both increases and decreases; using baseline_minus_current to capture declines as positive amounts'
    provider = FakeProvider([CalculationContract(**CALCULATION, unsupported_requirements=[claim]), CalculationContract(**CALCULATION)])
    verified, full = await computation(workspace, tmp_path, provider, program())
    assert verified.verified and full.data['metrics'] == {'region': 'Loss', 'decline': 200.0}
    assert verified.evidence['expected']['full_dataset_rows'] == 4
    assert [item['admitted'] for item in verified.evidence['interpretation_attempts']] == [False, True]
    assert len(provider.call_history) == 2
    assert 'Loss' not in json.loads(provider.call_history[1][-1]['content'])['interpretation_repair']['previous_contract'].values()


@pytest.mark.anyio
async def test_large_increase_cannot_pass_as_largest_decline(workspace, tmp_path):
    verified, full = await computation(workspace, tmp_path, FakeProvider([CalculationContract(**CALCULATION)]), program(absolute=True))
    assert full.data['metrics'] == {'region': 'Growth', 'decline': 1000.0}
    assert not verified.verified and verified.outcome == 'RECOVERABLE_FAILURE'
    assert verified.evidence['expected']['group'] == 'Loss'


@pytest.mark.anyio
@pytest.mark.parametrize('changes', [{'convention': 'current_minus_baseline'}, {'selection': 'min'}, {'baseline_period': '2026-09', 'current_period': '2026-08'}, {'measure': 'percent_change'}, {'unsupported_requirements': ['Forecast future revenue']}])
async def test_unusable_calculation_never_recomputes_or_auto_passes(workspace, tmp_path, changes):
    contract = CalculationContract(**{**CALCULATION, **changes})
    provider = FakeProvider([contract, deepcopy(contract)])
    verified, _ = await computation(workspace, tmp_path, provider, program())
    assert not verified.verified and verified.outcome == 'FATAL_FAILURE'
    assert verified.evidence['interpretation_failure']['origin'] == 'verifier_interpretation'
    assert 'expected' not in verified.evidence and len(provider.call_history) == 2


@pytest.mark.anyio
async def test_table_format_feedback_preserves_semantics_and_attempt_budget(workspace):
    capability = PythonCapability(run_id='result-repair')
    assert capability.profile_dataset('sales.csv').ok
    code = program().replace('totals.reset_index()', "totals.reset_index().to_dict(orient='records')")
    failed = await capability.execute_python(['sales.csv'], code)
    assert not failed.ok and failed.error_code == 'PYTHON_RESULT_ERROR'
    assert failed.data['repair_guidance']['scope'] == 'result_format'
    assert 'direction' in failed.data['repair_guidance']['instruction']
    assert failed.data['attempts_remaining'] == 2
    task = Subtask(task_id='data', goal=DATASET, success_criteria=[DATASET], verification_capability='python')
    state = GraphState(run_id='result-repair', objective=DATASET, tasks=[task], observation=failed.model_dump())
    prompt = decision_prompt(state, task, ContextMemory(), {})
    supplied = json.loads(prompt[-1]['content'].split('\n', 1)[1])['untrusted_task_data']['recent_outcomes']['observation']['data']
    assert supplied['repair_guidance'] == failed.data['repair_guidance']
    repaired = await capability.execute_python(['sales.csv'], program())
    assert repaired.ok and repaired.data['metrics'] == {'region': 'South', 'decline': 27000.0}
    assert repaired.data['attempt_number'] == 2 and repaired.data['attempts_remaining'] == 1
