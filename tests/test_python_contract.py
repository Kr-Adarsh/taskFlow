"""Generated-program contract checks; all model responses here are scripted."""
import json
from pathlib import Path

import pytest

from test_real_acceptance import real_workspace as real_workspace
from backend.app.agent.provider import FakeProvider
from backend.app.agent.verifier import snapshot_state, state_delta
from backend.app.agent_v2.graph import AgentRunner
from backend.app.capabilities.python import PythonCapability
from backend.app.capabilities.python.contracts import IMPORTS, RESULT_CONTRACT, result_issues
from backend.app.capabilities.python.safety import program_issues


BAD_CODE = """import os
df = pd.read_csv('sales.csv')
result = {'summary': 'Uncomputed', 'region': 'uncomputed', 'decline': 0}
"""
GOOD_CODE = """import pandas as pd
df = pd.read_csv(inputs['sales.csv'])
totals = df.groupby(['region', 'month'])['revenue'].sum().unstack()
declines = totals['2026-08'] - totals['2026-09']
result = {'summary': 'Compared full input',
          'metrics': {'region': str(declines.idxmax()), 'decline': float(declines.max()), 'rows': len(df)},
          'tables': {'comparison': totals.reset_index()}}
"""


@pytest.mark.anyio
async def test_multifault_preflight_never_launches_sandbox(real_workspace, monkeypatch):
    capability = PythonCapability(run_id='multifault')
    capability.profile_dataset('sales.csv')
    before = snapshot_state()
    async def forbidden(*args, **kwargs):
        pytest.fail('Preflight must reject before launching any sandbox')
    monkeypatch.setattr('backend.app.capabilities.python.isolated_stage', forbidden)
    result = await capability.execute_python(['sales.csv'], BAD_CODE)
    assert not result.ok and result.error_code == 'PYTHON_CONTRACT_ERROR'
    issues = result.evidence['issues']
    assert {issue['type'] for issue in issues} == {'unsupported_import', 'invalid_input_access', 'invalid_result_contract'}
    assert next(issue for issue in issues if issue['type'] == 'unsupported_import')['module'] == 'os'
    shape = next(issue for issue in issues if issue['type'] == 'invalid_result_contract')
    assert shape['missing'] == ['metrics'] and shape['unexpected'] == ['region', 'decline']
    contract = result.evidence['python_contract']
    assert contract['allowed_imports'] == sorted(IMPORTS)
    assert contract['available_inputs'] == ['sales.csv']
    assert contract['result_contract'] == RESULT_CONTRACT
    assert result.data == result.evidence
    assert result.evidence['repair_remaining'] is True
    assert result.evidence['repair_attempts_remaining'] == 2
    assert snapshot_state() == before
    assert not (real_workspace / 'artifacts').exists()


@pytest.mark.parametrize('code', [
    "pd.read_csv(inputs['sales.csv'])",
    "path = inputs['sales.csv']; pd.read_csv(path)",
    "import pandas as data; data.read_csv(inputs['sales.csv'])",
    "from pandas import read_csv; read_csv(inputs['sales.csv'])",
    "pd.read_csv('another.csv')",
])
def test_input_mapping_and_unknown_resources_are_not_false_positives(code):
    assert not program_issues(code, ['sales.csv'])


@pytest.mark.parametrize('expression', [
    "pd.read_csv('sales.csv')", "pd.read_parquet('sales.parquet')",
    "pd.read_json('x.json')", "pd.read_excel('x.xlsx')", "open('sales.csv')",
    "pd.read_csv(filepath_or_buffer='sales.csv')",
    "import pandas as data; data.read_csv('sales.csv')",
    "from pandas import read_csv; read_csv('sales.csv')",
])
def test_supplied_literal_paths_are_reported(expression):
    issues = program_issues(expression, ['sales.csv', 'sales.parquet', 'x.json', 'x.xlsx'])
    assert any(issue['type'] == 'invalid_input_access' for issue in issues)


@pytest.mark.parametrize('module', sorted(IMPORTS))
def test_existing_supported_imports_are_accepted(module):
    assert not program_issues(f'import {module}', [])


def test_import_configuration_is_shared(monkeypatch):
    from backend.app.capabilities.python import contracts, safety
    assert safety.IMPORTS is contracts.IMPORTS
    monkeypatch.setattr(safety, 'IMPORTS', safety.IMPORTS | {'decimal'})
    assert not program_issues('import decimal', [])


@pytest.mark.parametrize('code', [
    "result = {'summary': 'x', 'region': 'x'}", "result = {'metrics': []}",
    "result = {'summary': 5, 'metrics': {}}", "result = {'metrics': {'raw': [1, 2]}}",
    "result = {'metrics': {}, 'tables': []}", "result = {'metrics': {}, 'tables': {'table': [1]}}",
    "result = None",
])
def test_literal_invalid_result_is_rejected(code):
    assert any(issue['type'] == 'invalid_result_contract' for issue in program_issues(code, []))


@pytest.mark.parametrize('code', [
    "result = {'summary': 'Done', 'metrics': {}, 'tables': {}}",
    "result = {'summary': summary, 'metrics': {'value': computed}, 'tables': {'table': frame}}",
    "result = {'metrics': {}}",  # Summary remains optional in the existing worker contract.
    "result = build_result()",
    "result = {'metrics': []}; result['metrics'] = {}",
    "result = {'metrics': []}; fix(result)",
    "def fix():\n    result['metrics'] = {}\nresult = {'metrics': []}\nfix()",
    "result = {'metrics': []}; alias = result; alias['metrics'] = {}",
    "result = {'metrics': []}; result = {'metrics': {}}",
    "result = {'metrics': [], **fields}",
    "if condition:\n    result = {'metrics': []}\nelse:\n    result = build_result()",
])
def test_valid_and_uncertain_results_defer_without_false_rejection(code):
    assert not program_issues(code, [])


def test_static_and_concrete_validation_share_result_rules():
    code = "result = {'metrics': {'raw': [1]}}"
    assert program_issues(code, []) == result_issues({'metrics': {'raw': [1]}}, dataframe_type=())


def test_historical_flat_result_is_accepted_but_import_is_denied():
    path = Path('tests/fixtures/regressions/python/unsupported_result.py')
    issues = program_issues(path.read_text(), ['sales.csv'])
    assert {issue['type'] for issue in issues} == {'unsupported_import'}
    assert not any(issue['type'] == 'invalid_input_access' for issue in issues)


def test_historical_dynamic_flat_result_defers_but_literal_path_is_denied():
    path = Path('tests/fixtures/regressions/python/unsupported_access.py')
    issues = program_issues(path.read_text(), ['sales.csv'])
    assert {issue['type'] for issue in issues} == {'invalid_input_access'}


@pytest.mark.anyio
async def test_uncertain_result_still_uses_worker_validation(real_workspace):
    capability = PythonCapability(run_id='dynamic_result')
    capability.profile_dataset('sales.csv')
    code = "def make_result():\n    return {'metrics': []}\nresult = make_result()"
    assert not program_issues(code, ['sales.csv'])
    result = await capability.execute_python(['sales.csv'], code)
    assert not result.ok and result.data['stage'] == 'sample'
    assert 'Metrics' in result.error


@pytest.mark.anyio
async def test_exhaustion_remains_explicit_and_cannot_be_reset(real_workspace, monkeypatch):
    capability = PythonCapability(run_id='exhausted')
    capability.profile_dataset('sales.csv')
    async def forbidden(*args, **kwargs):
        pytest.fail('No bad or exhausted code may reach execution')
    monkeypatch.setattr('backend.app.capabilities.python.isolated_stage', forbidden)
    first = await capability.execute_python(['sales.csv'], BAD_CODE)
    second = await capability.execute_python(['sales.csv'], BAD_CODE + '\n# correction 1')
    third = await capability.execute_python(['sales.csv'], BAD_CODE + '\n# correction 2')
    capability.profile_dataset('sales.csv')
    exhausted = await capability.execute_python(['sales.csv'], GOOD_CODE)
    assert [first.data['attempts_remaining'], second.data['attempts_remaining'], third.data['attempts_remaining']] == [2, 1, 0]
    assert third.evidence['repair_remaining'] is False
    assert exhausted.error_code == 'CODE_REPAIR_EXHAUSTED'
    assert exhausted.data == exhausted.evidence
    assert exhausted.data['attempts_remaining'] == 0
    assert capability.calls == 3



@pytest.mark.anyio
async def test_one_model_repair_full_dataset_and_independent_verification(real_workspace, monkeypatch):
    from backend.app.capabilities.python import isolated_stage as original_stage
    stages = []
    async def observed(*args, **kwargs):
        result = await original_stage(*args, **kwargs)
        stages.append(result)
        return result
    monkeypatch.setattr('backend.app.capabilities.python.isolated_stage', observed)
    objective = 'Analyze sales.csv and identify the region with the largest absolute revenue decline between 2026-08 and 2026-09, including the calculated decline.'
    plan = {'objective': objective, 'success_criteria': [objective], 'tasks': [
        {'task_id': 'analysis', 'goal': objective, 'success_criteria': [objective], 'verification_capability': 'python'}]}
    def act(tool, **args):
        return {'thought': 'Exercise capability contract', 'action': 'act', 'tool_name': tool, 'tool_args': args}
    provider = FakeProvider([plan, act('profile_dataset', document_id='sales.csv'),
        act('execute_python', document_ids=['sales.csv'], code=BAD_CODE),
        act('execute_python', document_ids=['sales.csv'], code=GOOD_CODE),
        {'thought': 'Verify local computation', 'action': 'ready_for_verification',
         'result': {'metrics': {'region': 'South', 'decline': 27000.0, 'rows': 800}}},
        {'document_id': 'sales.csv', 'group_column': 'region', 'value_column': 'revenue',
         'aggregate': 'sum', 'period_column': 'month', 'baseline_period': '2026-08',
         'current_period': '2026-09', 'measure': 'difference', 'convention': 'baseline_minus_current',
         'selection': 'max', 'group_metric': 'region', 'value_metric': 'decline'}])
    prompts = []
    original_generate = provider.generate_structured
    async def capture(messages, schema, *args, **kwargs):
        prompts.append({'schema': schema.__name__, 'messages': messages})
        return await original_generate(messages, schema, *args, **kwargs)
    provider.generate_structured = capture
    before = snapshot_state()
    events = []
    result = await AgentRunner(provider=provider, event_callback=lambda run, kind, payload: events.append(
        {'event_type': kind, 'payload': payload})).execute_task(objective)
    assert result['status'] == 'completed', result
    assert not provider.responses
    assert [stage['stage'] for stage in stages] == ['sample', 'full']
    assert all(stage['ok'] and stage['isolation'] == {'namespaces': True, 'seccomp': True} for stage in stages)
    assert stages[-1]['metrics'] == {'region': 'South', 'decline': 27000.0, 'rows': 800}
    assert result['verification']['verified']
    expected = result['verification']['evidence']['expected']
    assert expected['full_dataset_rows'] == 800 and expected['group'] == 'South' and expected['value'] == 27000
    delta = state_delta(before, snapshot_state())
    assert not any(rows for changes in delta.values() for rows in changes.values())
    observations = [e['payload'] for e in events if e['event_type'] == 'OBSERVATION']
    assert [o.get('error_code') for o in observations] == [None, 'PYTHON_CONTRACT_ERROR', None]
    assert observations[1]['evidence']['repair_attempts_remaining'] == 2
    # The production prompt drops evidence; all repair details must survive in data.
    decision_prompts = [prompt for prompt in prompts if prompt['schema'] == 'Decision']
    repair_context = json.loads(decision_prompts[2]['messages'][-1]['content'].split('\n', 1)[1])
    visible = repair_context['untrusted_task_data']['recent_outcomes']['observation']['data']
    assert visible['issues'] == observations[1]['data']['issues']
    assert visible['python_contract'] == observations[1]['data']['python_contract']
    assert visible['repair_remaining'] is True
    assert observations[-1]['data']['inputs'][0]['rows'] == 800
    artifacts = result['report']['artifacts']
    assert len(artifacts) == 1
    assert (real_workspace / 'artifacts' / result['run_id']).is_dir()
    evidence_dir = Path('docs/validation/python-contract-preflight-20261006')
    evidence_dir.mkdir(exist_ok=True)
    (evidence_dir / 'scripted-regression.json').write_text(json.dumps(
        {'result': result, 'events': events, 'stages': stages, 'model_prompts': prompts, 'pre_state': before,
         'post_state': snapshot_state(), 'state_delta': delta}, indent=2) + '\n')
