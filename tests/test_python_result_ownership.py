"""Python completion uses execution evidence; model echoes are diagnostic only."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from test_real_acceptance import real_workspace as real_workspace
from backend.app.agent.provider import FakeProvider
from backend.app.agent.verifier import snapshot_state, state_delta
from backend.app.agent_v2.graph import AgentRunner, python_model_result_matches
from backend.app.agent_v2.verifier import CapabilityVerifier
from backend.app.capabilities.registry import build_registry
from backend.app.tools.base import ToolResult

OBJECTIVE = 'Analyze sales.csv and identify the region with the largest absolute revenue decline between 2026-08 and 2026-09, including the calculated decline.'
CONTRACT = {'document_id':'sales.csv','group_column':'region','value_column':'revenue','aggregate':'sum',
    'period_column':'month','baseline_period':'2026-08','current_period':'2026-09','measure':'difference',
    'convention':'baseline_minus_current','selection':'max','group_metric':'region','value_metric':'decline'}


def task(task_id='analysis', dependencies=()):
    return {'task_id':task_id,'goal':OBJECTIVE,'verification_capability':'python',
            'dependencies':list(dependencies),'success_criteria':[OBJECTIVE]}


def plan(tasks=None):
    return {'objective':OBJECTIVE,'success_criteria':[OBJECTIVE],'tasks':tasks or [task()]}


def act(tool, **args):
    return {'thought':'Exercise trusted-result ownership','action':'act','tool_name':tool,'tool_args':args}


def ready(result):
    return {'thought':'Request verification of execution','action':'ready_for_verification','result':result}


class RecordedProvider(FakeProvider):
    def __init__(self, responses):
        super().__init__(responses)
        self.schemas = []

    async def generate_structured(self, messages, response_schema, *args, **kwargs):
        self.schemas.append(response_schema.__name__)
        return await super().generate_structured(messages, response_schema, *args, **kwargs)


@pytest.mark.anyio
@pytest.mark.parametrize('echo,expected_match', [
    ({'region':'South','decline':27000.0}, True),
    ({}, 'unknown'),
    ({'region':'North','decline':1}, False),
    ({'metrics':{'region':'South','decline':27000}}, True),
])
async def test_full_python_result_survives_flat_empty_and_wrong_echo(real_workspace, monkeypatch, echo, expected_match):
    code = Path('tests/fixtures/regressions/python/analysis.py').read_text()
    provider = RecordedProvider([plan(),act('profile_dataset',document_id='sales.csv'),
        act('execute_python',document_ids=['sales.csv'],code=code),ready(echo),CONTRACT])
    audits = []; received = []
    original_audit = CapabilityVerifier.audit_mutations
    original_verify = CapabilityVerifier.verify_computation
    def audit(*args):
        result = original_audit(*args)
        audits.append(result.model_dump())
        return result
    async def verify(verifier, objective, subtask, *args, **kwargs):
        received.append(deepcopy(subtask.result))
        return await original_verify(verifier, objective, subtask, *args, **kwargs)
    monkeypatch.setattr(CapabilityVerifier,'audit_mutations',staticmethod(audit))
    monkeypatch.setattr(CapabilityVerifier,'verify_computation',verify)
    before = snapshot_state(); events = []
    runner = AgentRunner(provider=provider,event_callback=lambda run,kind,payload:events.append({'event_type':kind,'payload':payload}))
    result = await runner.execute_task(OBJECTIVE)
    canonical = {'summary':'','metrics':{'region':'South','decline':27000.0},'tables':{}}
    assert result['status']=='completed',result
    assert not provider.responses
    assert provider.schemas==['TaskPlan','Decision','Decision','Decision','CalculationContract']
    assert received==[canonical]
    assert result['verification']['verified']
    assert result['verification']['evidence']['expected']['full_dataset_rows']==800
    assert result['verification']['evidence']['expected']['group']=='South'
    assert result['verification']['evidence']['expected']['value']==27000
    assert len(audits)==1 and audits[0]['verified']
    final = result['report']['tasks'][0]
    assert final['result']['metrics']==canonical['metrics']
    assert final['result']['summary']=='' and final['result']['tables']=={}
    assert final['result']['verification']['verified']
    assert final['model_reported_result']==echo
    assert final['model_result_matches_authoritative']==expected_match
    authority = final['authoritative_python_result']
    assert authority['task_id']=='analysis' and authority['run_id']==result['run_id']
    assert authority['tool']=='execute_python' and authority['capability']=='python' and authority['stage']=='full'
    assert authority['canonical_result']==canonical and 'verification' not in authority['canonical_result']
    assert authority['code_sha256']==runner.python_results['analysis']['code_sha256']
    assert authority['inputs'][0]['rows']==800
    assert authority['input_hashes']['sales.csv']==authority['inputs'][0]['sha256']
    assert authority['artifacts']==[]
    delta = state_delta(before,snapshot_state())
    assert not any(rows for changes in delta.values() for rows in changes.values())
    assert next(e['payload'] for e in events if e['event_type']=='PYTHON_RESULT_BINDING')['model_result_matches_authoritative']==expected_match
    if echo=={'region':'South','decline':27000.0}:
        out=Path('docs/validation/python-result-ownership-20261006')
        (out/'exact-failure-regression.json').write_text(json.dumps({'result':result,'received_by_verifier':received,
            'audits':audits,'events':events,'schemas':provider.schemas,'pre_state':before,'post_state':snapshot_state(),'state_delta':delta},indent=2)+'\n')


@pytest.mark.anyio
async def test_premature_ready_cannot_invent_python_result(real_workspace):
    provider=RecordedProvider([plan(),ready({'region':'South','decline':27000})])
    runner=AgentRunner(provider=provider)
    result=await runner.execute_task(OBJECTIVE)
    assert result['status']=='failed' and result['error_code']=='PYTHON_RESULT_NOT_READY'
    assert provider.schemas==['TaskPlan','Decision'] and not provider.responses
    final=result['report']['tasks'][0]
    assert final['result']=={} and final['authoritative_python_result'] is None
    assert final['model_reported_result']=={'region':'South','decline':27000}
    assert final['errors'][0]['code']=='PYTHON_RESULT_NOT_READY'
    assert final['verification_attempts']==0 and not runner.python_results


@pytest.mark.anyio
@pytest.mark.parametrize('stage,ok', [('sample',True),('full',False)])
async def test_sample_or_failed_full_never_owns_result(real_workspace,stage,ok):
    registry=build_registry()
    async def partial(*args,**kwargs):
        return ToolResult(ok=ok,data={'ok':ok,'stage':stage,'summary':'','metrics':{'region':'South','decline':27000},'tables':{}},
                          error=None if ok else 'Full execution failed',error_code=None if ok else 'PYTHON_RUNTIME_ERROR')
    registry.get('execute_python').func=partial
    provider=RecordedProvider([plan(),act('execute_python',document_ids=['sales.csv'],code='result={}'),ready({'metrics':{'region':'South','decline':27000}})])
    runner=AgentRunner(provider=provider,tool_registry=registry)
    result=await runner.execute_task(OBJECTIVE)
    assert result['status']=='failed' and result['error_code']=='PYTHON_RESULT_NOT_READY'
    assert provider.schemas==['TaskPlan','Decision','Decision']
    assert not provider.responses and not runner.python_results
    assert result['report']['tasks'][0]['authoritative_python_result'] is None
    assert result['report']['tasks'][0]['result']=={}


@pytest.mark.anyio
async def test_completed_task_a_never_supplies_task_b_result(real_workspace):
    code="df=pd.read_csv(inputs['sales.csv'])\ntotals=df.groupby(['region','month'])['revenue'].sum().unstack()\ndecline=totals['2026-08']-totals['2026-09']\nresult={'region':decline.idxmax(),'decline':decline.max()}"
    provider=RecordedProvider([plan([task('a'),task('b',['a'])]),act('profile_dataset',document_id='sales.csv'),
        act('execute_python',document_ids=['sales.csv'],code=code),ready({}),CONTRACT,
        ready({'region':'South','decline':27000})])
    runner=AgentRunner(provider=provider)
    result=await runner.execute_task(OBJECTIVE)
    assert result['status']=='failed' and result['error_code']=='PYTHON_RESULT_NOT_READY'
    assert not provider.responses and provider.schemas.count('CalculationContract')==1
    a,b=result['report']['tasks']
    assert a['status']=='COMPLETED' and a['authoritative_python_result']['task_id']=='a'
    assert a['result']['verification']['verified']
    assert b['authoritative_python_result'] is None and b['result']=={} and b['verification_attempts']==0
    assert set(runner.python_results)=={'a'}


def test_diagnostic_comparison_does_not_guess_or_equate_boolean_and_numeric():
    canonical={'summary':'','metrics':{'value':1},'tables':{}}
    assert python_model_result_matches({},canonical)=='unknown'
    assert python_model_result_matches({'metrics':{'value':True}},canonical) is False
    assert python_model_result_matches({'metrics':{'value':[1]}},canonical)=='unknown'
    assert python_model_result_matches({'value':float('nan')},canonical)=='unknown'
    assert python_model_result_matches({'value':1.0},canonical) is True
    assert python_model_result_matches({'summary':'Unverified narrative','metrics':{'value':1}},canonical) is False
