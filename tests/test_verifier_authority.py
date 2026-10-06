"""Offline authority regressions using the frozen successful business mutation."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from backend.app.agent.provider import FakeProvider
from backend.app.agent.schemas import SummaryAssessment, VerificationIntent
from backend.app.agent.verifier import VerifierEngine, snapshot_state, state_delta
from backend.app.agent_v2.context import ContextMemory
from backend.app.agent_v2.graph import AgentRunner
from backend.app.agent_v2.state import GraphState, Subtask
from backend.app.agent_v2.verifier import CapabilityVerifier
from backend.app.workspace.db import get_db_connection
from backend.app.workspace.seed import seed_workspace

pytestmark = pytest.mark.anyio
HISTORY = Path('tests/fixtures/regressions/ticket_verification')
EVIDENCE = Path('docs/validation/verifier-authority-20261006')


class SchemaProvider(FakeProvider):
    def __init__(self, responses):
        super().__init__(responses)
        self.schemas = []

    async def generate_structured(self, messages, schema, *args, **kwargs):
        self.schemas.append(schema.__name__)
        return await super().generate_structured(messages, schema, *args, **kwargs)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    path = tmp_path / 'authority.db'
    monkeypatch.setenv('TASKFLOW_DB_PATH', str(path))
    seed_workspace(path, force_reseed=True)
    return path


def historical_run():
    return json.loads((HISTORY / 'complaint.json').read_text())


def frozen_intent():
    requests = json.loads((HISTORY / 'complaint-http.json').read_text())
    request = next(item for item in requests if item['schema']['title'] == 'VerificationIntent')
    return VerificationIntent.model_validate_json(request['response']['choices'][0]['message']['content'])


def persist_frozen_ticket(path):
    record = historical_run()
    assert snapshot_state(path) == record['pre_state']
    row, = record['state_delta']['tickets']['created']
    columns = list(row)
    with get_db_connection(path) as connection:
        connection.execute(
            'INSERT INTO support_tickets (' + ','.join(columns) + ') VALUES (' + ','.join('?' for _ in columns) + ')',
            [row[column] for column in columns],
        )
        connection.commit()
    assert snapshot_state(path) == record['post_state']
    return row


def assessment():
    return SummaryAssessment(accurate=True, reason='Summary describes the reported outage and escalation request',
        source_quotes=['severe database synchronization outages across our EU servers',
                       'Need urgent resolution and senior engineering support immediately.'], contradictions=[])


def browser_task(objective):
    return Subtask(task_id='outcome', goal=objective, success_criteria=[objective], verification_capability='browser')


async def test_exact_frozen_complaint_reaches_all_independent_checks(workspace, monkeypatch):
    record = historical_run()
    row = persist_frozen_ticket(workspace)
    provider = SchemaProvider([frozen_intent(), assessment()])
    verifier = CapabilityVerifier(provider, workspace)
    ticket_check = AsyncMock(wraps=verifier.browser._verify_ticket)
    mutation_check = Mock(wraps=verifier.browser._check_delta)
    monkeypatch.setattr(verifier.browser, '_verify_ticket', ticket_check)
    monkeypatch.setattr(verifier.browser, '_check_delta', mutation_check)
    task = browser_task(record['objective'])
    result = await verifier.verify(record['objective'], task, record['pre_state'], ContextMemory())
    assert result.outcome == 'PASS' and result.verified
    ticket_check.assert_awaited_once()
    mutation_check.assert_called_once()
    assert provider.schemas == ['VerificationIntent', 'SummaryAssessment'] and not provider.responses
    checks = {item['criterion']: item for item in result.evidence['verification_result']['criteria_results']}
    resolved = checks['Source customer and exact CRM account resolved']['evidence']
    assert resolved == {'source_id': 'complaint_4821.txt', 'customer': 'Acme Corp', 'account_id': 1, 'tier': 'Enterprise'}
    assert checks['Exactly one complaint-linked ticket']['evidence']['count'] == 1
    assert checks['Requested ticket created during run']['evidence'] == {'record_id': row['id'], 'created_during_run': True}
    assert checks['Requested ticket priority']['evidence']['actual'] == 'High'
    assert checks['Grounded complaint summary']['passed']
    assert checks['Grounded complaint summary']['evidence']['summary'] == row['summary']
    assert checks['Grounded complaint summary']['evidence']['source_id'] == row['source_reference']
    assert checks['No unwanted mutations']['passed']
    assert all(item['passed'] for item in checks.values())
    context = result.evidence['verification_result']['context']
    assert context['interpreted_contract']['unsupported_criteria'] == frozen_intent().unsupported_criteria
    assert context['interpretation_diagnostics'] == {
        'unsupported_criteria': frozen_intent().unsupported_criteria,
        'authority': 'diagnostic_only', 'evaluated_as_criteria': False,
    }
    assert not any(item['criterion'] in frozen_intent().unsupported_criteria for item in checks.values())
    assert 'diagnostics' in result.summary and context['interpretation_failure'] is None
    task.result['verification'] = result.model_dump()
    audit = verifier.audit_mutations([task], record['pre_state'], snapshot_state(workspace))
    assert audit.verified
    summary_data = json.loads(provider.call_history[1][-1]['content'])
    assert summary_data['summary'] == row['summary'] and 'Customer Name: Acme Corp' in summary_data['source_text']
    (EVIDENCE / 'frozen-complaint-regression.json').write_text(json.dumps({
        'intent': frozen_intent().model_dump(), 'exact_ticket': row, 'schemas': provider.schemas,
        'summary_request': provider.call_history[1], 'verification': result.model_dump(), 'final_audit': audit.model_dump(),
        'pre_state': record['pre_state'], 'post_state': snapshot_state(workspace),
        'state_delta': state_delta(record['pre_state'], snapshot_state(workspace)),
    }, indent=2) + '\n')


@pytest.mark.parametrize('defect', ['priority', 'source', 'unwanted_mutation', 'summary'])
async def test_diagnostic_claims_cannot_override_failed_application_checks(workspace, defect):
    record = historical_run()
    persist_frozen_ticket(workspace)
    if defect != 'summary':
        statements = {
            'priority': "UPDATE support_tickets SET priority='Low' WHERE ticket_id='TIK-8402'",
            'source': "UPDATE support_tickets SET source_reference='unrelated.txt' WHERE ticket_id='TIK-8402'",
            'unwanted_mutation': "UPDATE crm_accounts SET status='Inactive' WHERE customer_name='Globex Inc'",
        }
        with get_db_connection(workspace) as connection:
            connection.execute(statements[defect])
            connection.commit()
    semantic = assessment() if defect != 'summary' else SummaryAssessment(
        accurate=False, reason='Unsupported summary', source_quotes=[], contradictions=['Invented claim'])
    provider = SchemaProvider([frozen_intent(), semantic])
    result = await CapabilityVerifier(provider, workspace).verify(record['objective'], browser_task(record['objective']),
        record['pre_state'], ContextMemory())
    assert not result.verified and result.outcome == 'RECOVERABLE_FAILURE'
    assert result.evidence['verification_result']['context']['interpretation_failure'] is None
    assert result.evidence['verification_result']['context']['interpretation_diagnostics']['unsupported_criteria']
    expected = {'priority': 'Requested ticket priority', 'source': 'Exactly one complaint-linked ticket',
                'unwanted_mutation': 'No unwanted mutations', 'summary': 'Grounded complaint summary'}[defect]
    assert any(item['criterion'] == expected and not item['passed']
               for item in result.evidence['verification_result']['criteria_results'])


@pytest.mark.parametrize('collection', ['invoices', 'accounts'])
async def test_supported_families_preserve_arbitrary_diagnostics_without_claiming_them(workspace, collection):
    before = snapshot_state(workspace)
    if collection == 'invoices':
        from backend.app.workspace.models import InvoiceCreate
        from backend.app.workspace.service import create_invoice
        create_invoice(InvoiceCreate(company='Acme Corp', invoice_number='INV-1044', amount='84500', currency='INR',
            due_date='2026-10-15', source_reference='acme_invoice_1044.pdf'), workspace)
        intent = VerificationIntent(collection=collection, company='Acme Corp', selection='latest')
    else:
        intent = VerificationIntent(collection=collection, company='Acme Corp', requested_fields=['tier'], require_new_record=False)
    intent.unsupported_criteria = ['Model capability claim with no execution authority']
    task = browser_task('Evaluate supported persisted fields')
    if collection == 'accounts':
        task.result = {'customer_name': 'Acme Corp', 'tier': 'Enterprise'}
    result = await CapabilityVerifier(SchemaProvider([intent]), workspace).verify(task.goal, task, before, ContextMemory())
    assert result.verified and 'diagnostics' in result.summary
    assert result.evidence['verification_result']['context']['interpretation_diagnostics']['evaluated_as_criteria'] is False


@pytest.mark.parametrize('intent,outcome', [
    (VerificationIntent(collection='unsupported', unsupported_criteria=['Sending external email']), 'FATAL_FAILURE'),
    (VerificationIntent(collection='invoices', company='Acme Corp', selection='unresolved'), 'FATAL_FAILURE'),
    (VerificationIntent(collection='tickets'), 'FATAL_FAILURE'),
    (VerificationIntent(collection='tickets', complaint_id='4821', requested_fields=['mrr']), 'FATAL_FAILURE'),
])
async def test_unusable_interpretation_stops_graph_without_executor_retry(workspace, intent, outcome, monkeypatch):
    objective = 'Resolve the requested outcome'
    plan = {'objective': objective, 'success_criteria': [objective], 'tasks': [{
        'task_id': 'outcome', 'goal': objective, 'success_criteria': [objective], 'verification_capability': 'browser'}]}
    provider = SchemaProvider([plan, {'thought': 'Request independent verification', 'action': 'ready_for_verification'}, intent, deepcopy(intent)])
    events = []
    runner = AgentRunner(provider=provider, db_path=workspace,
        event_callback=lambda _, kind, payload: events.append((kind, payload)))
    ticket_check = AsyncMock(side_effect=AssertionError('Unusable intent entered ticket verification'))
    monkeypatch.setattr(runner.capability_verifier.browser, '_verify_ticket', ticket_check)
    result = await runner.execute_task(objective)
    assert result['status'] == ('waiting_for_clarification' if outcome == 'AMBIGUITY' else 'failed')
    assert next(payload for kind, payload in events if kind == 'VERIFICATION')['outcome'] == outcome
    assert result['steps'] == 1 and result['report']['tasks'][0]['verification_attempts'] == 1
    assert provider.schemas == ['TaskPlan', 'Decision', 'VerificationIntent', 'VerificationIntent'] and not provider.responses
    assert runner.capability_verifier.browser.intent is None
    assert runner.capability_verifier.browser._intent_objective is None
    ticket_check.assert_not_awaited()


@pytest.mark.parametrize('intent', [
    VerificationIntent(collection='unsupported'),
    VerificationIntent(collection='invoices', company='Acme Corp', selection='unresolved'),
    VerificationIntent(collection='invoices', company='Acme Corp', selection='specific'),
    VerificationIntent(collection='accounts', company='Acme Corp'),
])
async def test_unusable_intent_is_not_cached_as_a_reusable_interpretation(workspace, intent):
    valid = frozen_intent()
    provider = SchemaProvider([deepcopy(intent), valid])
    verifier = VerifierEngine(workspace, provider=provider)
    assert await verifier.interpret('Unchanged objective', []) == valid
    assert verifier.interpretation_attempts[0]['admitted'] is False
    assert verifier.interpretation_attempts[1]['admitted'] is True
    assert verifier.intent == valid
    assert await verifier.interpret('Unchanged objective', ['Different planner wording']) == valid
    assert provider.schemas == ['VerificationIntent', 'VerificationIntent']


async def test_missing_ticket_is_recoverable_and_workspace_change_reuses_valid_intent(workspace, monkeypatch):
    record = historical_run()
    provider = SchemaProvider([frozen_intent(), assessment()])
    runner = AgentRunner(provider=provider, db_path=workspace)
    task = browser_task(record['objective'])
    state = GraphState(run_id='offline-recovery', objective=record['objective'], tasks=[task], current_task_id=task.task_id)
    runner.task_pre_states = {task.task_id: snapshot_state(workspace)}
    runner.memory_mgr = ContextMemory()
    runner.python_results = {}
    monkeypatch.setattr(runner, '_update_run_status', lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, '_emit_event', lambda *args, **kwargs: None)
    first = await runner.verify(state)
    assert first['verification'].outcome == 'RECOVERABLE_FAILURE' and first['route'] == 'context_build'
    assert any('No support ticket' in item for item in first['verification'].discrepancies)
    assert provider.schemas == ['VerificationIntent']
    persist_frozen_ticket(workspace)
    second = await runner.verify(state)
    assert second['verification'].verified and second['route'] == 'select_subtask'
    assert provider.schemas == ['VerificationIntent', 'SummaryAssessment'] and not provider.responses
    assert runner.capability_verifier.browser.intent == frozen_intent()
    assert task.verification_attempts == 2


def test_unknown_constructed_collection_is_rejected_before_application_checks():
    from backend.app.agent.verifier import InterpretationFailure
    with pytest.raises(InterpretationFailure) as error:
        VerifierEngine._validate_intent(VerificationIntent.model_construct(collection='contacts'))
    assert error.value.outcome == 'FATAL_FAILURE'
