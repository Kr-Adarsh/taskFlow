"""Trusted result representations and bounded Python iteration; no live model calls."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_real_acceptance import real_workspace as real_workspace
from backend.app.agent.provider import FakeProvider
from backend.app.agent.verifier import snapshot_state, state_delta
from backend.app.agent_v2.graph import AgentRunner
from backend.app.capabilities.python import PythonCapability
from backend.app.capabilities.python.contracts import canonicalize_result, ResultError
from backend.app.capabilities.python.safety import program_issues
from backend.app.capabilities.python.sandbox import isolated_stage

OUT = Path('docs/validation/python-robustness-20261006')


def normalize(result):
    return canonicalize_result(result, np=np, pd=pd)


@pytest.mark.parametrize('value,expected', [
    (np.int64(27000), 27000), (np.uint64(2**63 + 1), 2**63 + 1),
    (np.float32(1.25), 1.25), (np.float64(1.25), 1.25), (np.bool_(True), True),
    (np.longdouble(1.25), 1.25),
    (np.str_('South'), 'South'), (np.datetime64('2026-10-06'), '2026-10-06'),
    (datetime.date(2026, 10, 6), '2026-10-06'),
    (datetime.datetime(2026, 10, 6, 12, 30), '2026-10-06T12:30:00'),
    (pd.Timestamp('2026-10-06T12:30:00.123456789'), '2026-10-06T12:30:00.123456789'),
    (pd.Timestamp('2026-10-06T12:30:00Z'), '2026-10-06T12:30:00+00:00'),
    (pd.NA, None), (pd.NaT, None), (np.datetime64('NaT', 'ns'), None),
])
def test_safe_scalar_representations_preserve_json_values(value, expected):
    result = normalize({'metrics': {'value': value}})
    assert result == {'summary': '', 'metrics': {'value': expected}, 'tables': {}}
    encoded = json.dumps(result, allow_nan=False)
    assert json.loads(encoded)['metrics']['value'] == expected
    assert type(result['metrics']['value']) is type(expected)


def test_flat_numpy_metrics_use_only_canonical_envelope():
    result = normalize({'region': np.str_('South'), 'decline': np.int64(27000)})
    assert json.dumps(result, separators=(',', ':')) == '{"summary":"","metrics":{"region":"South","decline":27000},"tables":{}}'
    assert not program_issues("result = {'region': 'South', 'decline': 27000}", [])


def test_extended_precision_is_not_rounded_for_json():
    value = np.longdouble('0.1234567890123456789')
    if value == np.longdouble(float(value)):
        pytest.skip('This platform has no additional long-double precision')
    with pytest.raises(ResultError, match='precision'):
        normalize({'metrics': {'value': value}})


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf, np.float32(np.nan), np.float64(np.inf)])
def test_nonfinite_numbers_are_rejected_with_path_and_original_type(value):
    with pytest.raises(ResultError) as error:
        normalize({'metrics': {'decline': value}})
    issue = error.value.issues[0]
    assert issue['path'] == 'result.metrics.decline'
    assert issue['actual_type'] == type(value).__module__ + '.' + type(value).__name__
    assert issue['expected'] == 'finite JSON numeric scalar'


@pytest.mark.parametrize('value', [np.array([1]), pd.Series([1]), [1], {'nested': 1}, np.complex64(1+2j), b'bytes'])
def test_array_series_nested_and_unsupported_values_are_not_normalized(value):
    with pytest.raises(ResultError) as error:
        normalize({'metrics': {'decline': value}})
    assert error.value.issues[0]['path'] == 'result.metrics.decline'
    assert 'scalar' in error.value.issues[0]['expected']


def test_custom_conversion_methods_are_never_called():
    class Custom:
        def item(self):
            pytest.fail('Arbitrary item conversion is forbidden')
        def isoformat(self):
            pytest.fail('Arbitrary date conversion is forbidden')
        def __str__(self):
            pytest.fail('Arbitrary string conversion is forbidden')
    with pytest.raises(ResultError):
        normalize({'metrics': {'value': Custom()}})


def test_custom_timezone_methods_are_never_called():
    class CustomTimezone(datetime.tzinfo):
        def utcoffset(self, value):
            pytest.fail('Arbitrary timezone conversion is forbidden')
    value = datetime.datetime(2026, 10, 6, tzinfo=CustomTimezone())
    with pytest.raises(ResultError):
        normalize({'metrics': {'value': value}})


@pytest.mark.parametrize('result', [
    {'summary': 'x', 'region': 'South'}, {'metrics': {}, 'decline': 27000},
    {'tables': {}, 'region': 'South'}, {'summary': pd.NA, 'metrics': {}},
    {'metrics': {}, 'tables': {'raw': pd.Series([1])}},
])
def test_reserved_mixed_shapes_and_disallowed_nulls_are_rejected(result):
    with pytest.raises(ResultError):
        normalize(result)


@pytest.mark.anyio
@pytest.mark.parametrize('flat', [False, True])
async def test_numpy_int64_serializes_inside_actual_worker(real_workspace, flat):
    capability = PythonCapability(run_id='scalar_wire_' + str(flat))
    capability.profile_dataset('sales.csv')
    metrics = "{'region': np.str_('South'), 'decline': np.int64(27000)}"
    code = 'result = ' + (metrics if flat else "{'summary': 'Measured', 'metrics': " + metrics + '}')
    result = await capability.execute_python(['sales.csv'], code)
    assert result.ok, result.error
    assert result.data['metrics'] == {'region': 'South', 'decline': 27000}
    assert type(result.data['metrics']['decline']) is int
    assert result.data['summary'] == ('' if flat else 'Measured')
    assert result.data['tables'] == {}
    assert result.data['isolation'] == {'namespaces': True, 'seccomp': True}
    assert result.data['inputs'][0]['rows'] == 800
    assert '"decline": 27000' in json.dumps(result.data)


@pytest.mark.anyio
async def test_runtime_result_security_timeout_and_output_categories(real_workspace, tmp_path):
    source = Path('fixtures/datasets/sales.csv')
    cases = [
        ("raise ValueError('analysis failed')", 'PYTHON_RUNTIME_ERROR'),
        ("result={'metrics': {'decline': np.inf}}", 'PYTHON_RESULT_ERROR'),
        ("result={'metrics': {'decline': pd.Series([1])}}", 'PYTHON_RESULT_ERROR'),
        ("pd.DataFrame({'x':[1]}).to_csv(inputs['sales.csv'])\nresult={}", 'PYTHON_SECURITY_DENIED'),
    ]
    for index, (code, expected) in enumerate(cases):
        result = await isolated_stage(code, {'sales.csv': source}, tmp_path / str(index), 'sample')
        assert not result['ok'] and result['category'] == expected, result
        if expected == 'PYTHON_RESULT_ERROR':
            assert result['issues'][0]['path'] == 'result.metrics.decline'
    timeout = await isolated_stage('while True: pass', {'sales.csv': source}, tmp_path/'timeout', 'sample', timeout=1)
    assert timeout['category'] == 'PYTHON_TIMEOUT'
    output = await isolated_stage("while True: pd.io.common.os.write(1,b'x'*1048576)", {'sales.csv': source}, tmp_path/'flood', 'sample')
    assert output['category'] == 'PYTHON_OUTPUT_LIMIT'


@pytest.mark.anyio
async def test_identical_failed_program_is_no_progress_without_spending_attempt(real_workspace):
    capability = PythonCapability(run_id='same_code')
    capability.profile_dataset('sales.csv')
    code = "raise ValueError('same failure')"
    first = await capability.execute_python(['sales.csv'], code)
    for _ in range(4):
        capability.profile_dataset('sales.csv')
        repeated = await capability.execute_python(['sales.csv'], code)
        assert repeated.error_code == 'REDUNDANT_ACTION'
        assert repeated.data['no_progress'] and repeated.data['no_op']
        assert repeated.data['category'] == 'PYTHON_RUNTIME_ERROR'
        assert repeated.data['attempt_number'] == 1 and repeated.data['attempts_remaining'] == 2
    assert first.data['attempt_number'] == 1 and capability.calls == 1
    repaired = await capability.execute_python(['sales.csv'], "result={'rows':len(pd.read_csv(inputs['sales.csv']))}")
    assert repaired.ok and repaired.data['attempt_number'] == 2


@pytest.mark.anyio
async def test_attempts_are_per_task_not_input_or_profile(real_workspace):
    capability = PythonCapability(run_id='task_budget')
    capability.profile_dataset('sales.csv')
    capability.task_id = 'one'
    for attempt in range(1, 4):
        failed = await capability.execute_python(['sales.csv'], f"raise ValueError('failure {attempt}')")
        assert failed.data['attempt_number'] == attempt
    capability.profile_dataset('sales.csv')
    exhausted = await capability.execute_python(['sales.csv'], "result={'x':1}")
    assert exhausted.error_code == 'CODE_REPAIR_EXHAUSTED'
    capability.task_id = 'two'
    result = await capability.execute_python(['sales.csv'], "result={'x':1}")
    assert result.ok and result.data['attempt_number'] == 1


@pytest.mark.anyio
async def test_identical_failure_reaches_existing_graph_no_progress_guard(real_workspace):
    objective = 'Count sales.csv rows'
    code = "raise ValueError('unchanged failure')"
    plan = {'objective': objective, 'success_criteria': [objective], 'tasks': [
        {'task_id':'analysis','goal':objective,'verification_capability':'python','success_criteria':[objective]}]}
    def action(tool, **args):
        return {'thought':'Exercise no progress','action':'act','tool_name':tool,'tool_args':args}
    provider = FakeProvider([plan, action('profile_dataset',document_id='sales.csv'),
        *[action('execute_python',document_ids=['sales.csv'],code=code) for _ in range(4)]])
    runner = AgentRunner(provider=provider)
    result = await runner.execute_task(objective)
    assert result['status'] == 'failed' and 'No progress' in result['error']
    assert not provider.responses
    assert runner.registry.python.calls == 1


@pytest.mark.anyio
async def test_historical_corrected_program_now_runs_without_rewriting(real_workspace):
    path = Path('tests/fixtures/regressions/python/input_contract.py')
    code = path.read_text()
    capability = PythonCapability(run_id='historical_numpy')
    capability.profile_dataset('sales.csv')
    result = await capability.execute_python(['sales.csv'], code)
    assert result.ok, result.error
    assert result.data['metrics'] == {'region_with_max_decline': 'South', 'max_absolute_decline': 27000}
    assert type(result.data['metrics']['max_absolute_decline']) is int
    assert result.data['code_sha256'] == hashlib.sha256(code.encode()).hexdigest()
    assert result.data['inputs'][0]['rows'] == 800


@pytest.mark.anyio
async def test_scripted_two_corrections_full_data_and_independent_verifier(real_workspace, monkeypatch):
    from backend.app.capabilities.python import isolated_stage as original_stage
    stages = []
    async def observed(*args, **kwargs):
        result = await original_stage(*args, **kwargs)
        stages.append(result)
        return result
    monkeypatch.setattr('backend.app.capabilities.python.isolated_stage', observed)
    objective = 'Analyze sales.csv and identify the region with the largest absolute revenue decline between 2026-08 and 2026-09, including the calculated decline.'
    plan = {'objective': objective, 'success_criteria': [objective], 'tasks': [
        {'task_id': 'analysis', 'goal': objective, 'verification_capability': 'python', 'success_criteria': [objective]}]}
    bad = "result={'summary': 'mixed', 'region': 'x'}"
    runtime_error = "raise ValueError('calculation needs correction')"
    code = """df=pd.read_csv(inputs['sales.csv'])
totals=df.groupby(['region','month'])['revenue'].sum().unstack()
decline=totals['2026-08']-totals['2026-09']
result={'region':decline.idxmax(), 'decline':decline.max(), 'rows':len(df)}
"""
    def act(tool, **args):
        return {'thought': 'Exercise Python iteration', 'action': 'act', 'tool_name': tool, 'tool_args': args}
    provider = FakeProvider([plan, act('profile_dataset', document_id='sales.csv'),
        act('execute_python', document_ids=['sales.csv'], code=bad),
        act('execute_python', document_ids=['sales.csv'], code=runtime_error),
        act('execute_python', document_ids=['sales.csv'], code=code),
        {'thought': 'Verify full local result', 'action': 'ready_for_verification',
         'result': {'metrics': {'region': 'South', 'decline': 27000, 'rows': 800}}},
        {'document_id': 'sales.csv', 'group_column': 'region', 'value_column': 'revenue',
         'aggregate': 'sum', 'period_column': 'month', 'baseline_period': '2026-08', 'current_period': '2026-09',
         'measure': 'difference', 'convention': 'baseline_minus_current', 'selection': 'max',
         'group_metric': 'region', 'value_metric': 'decline'}])
    before = snapshot_state(); events = []; prompts = []
    generate = provider.generate_structured
    async def capture(messages, schema, *args, **kwargs):
        prompts.append({'schema': schema.__name__, 'messages': messages})
        return await generate(messages, schema, *args, **kwargs)
    provider.generate_structured = capture
    result = await AgentRunner(provider=provider, event_callback=lambda run, kind, payload: events.append(
        {'event_type': kind, 'payload': payload})).execute_task(objective)
    assert result['status'] == 'completed', result
    assert not provider.responses
    assert [s['stage'] for s in stages] == ['sample', 'sample', 'full']
    assert not stages[0]['ok'] and stages[1]['ok'] and stages[2]['ok']
    assert stages[-1]['metrics'] == {'region': 'South', 'decline': 27000, 'rows': 800}
    assert type(stages[-1]['metrics']['decline']) is int
    assert result['verification']['verified']
    expected = result['verification']['evidence']['expected']
    assert expected['full_dataset_rows'] == 800 and expected['group'] == 'South' and expected['value'] == 27000
    delta = state_delta(before, snapshot_state())
    assert not any(rows for changes in delta.values() for rows in changes.values())
    obs = [e['payload'] for e in events if e['event_type'] == 'OBSERVATION' and e['payload'].get('data', {}).get('attempt_number')]
    assert [o['data']['attempt_number'] for o in obs] == [1,2,3]
    assert [o['data']['attempts_remaining'] for o in obs] == [2,1,0]
    decisions = [p for p in prompts if p['schema'] == 'Decision']
    repair = json.loads(decisions[3]['messages'][-1]['content'].split('\n',1)[1])['untrusted_task_data']['recent_outcomes']['observation']['data']
    assert repair['attempt_number'] == 2 and repair['attempts_remaining'] == 1
    OUT.mkdir(exist_ok=True)
    (OUT/'scripted-dataset.json').write_text(json.dumps({'result':result,'events':events,'stages':stages,
        'prompts':prompts,'code':code,'pre_state':before,'post_state':snapshot_state(),'state_delta':delta},indent=2)+'\n')
