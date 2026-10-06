"""Honest terminal outcomes, persisted evidence, finite runs and SSE replay."""
import asyncio
import json
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.agent.loop import AgentRunner
from backend.app.agent.provider import FakeProvider
from backend.app.agent.schemas import TaskPlan, AgentDecision, VerificationIntent
from backend.app.agent.verifier import VerifierEngine
from backend.app.tools.registry import ToolRegistry
from backend.app.tools.base import ToolResult
from backend.app.workspace.db import get_db_connection
from backend.app.workspace.seed import reset_demo_env

@pytest.fixture(autouse=True)
def reset():
    reset_demo_env()


def provider(*decisions):
    return FakeProvider([TaskPlan(objective='Check',success_criteria=['Verified'],strategy=['Inspect']),*decisions])

@pytest.mark.anyio
async def test_clarification_report_memory_evidence_and_replay():
    registry=ToolRegistry()
    registry.register('inspect','Inspect',{},lambda: ToolResult(ok=False,error='Unclear',error_code='AMBIGUOUS',evidence={'source_id':'one.txt'}))
    fake=provider(AgentDecision(thought='Inspect',action='act',tool_name='inspect',tool_args={}),AgentDecision(thought='Unclear',action='need_clarification',clarification_question='Which source?'))
    runner=AgentRunner(provider=fake,tool_registry=registry)
    result=await runner.execute_task('Check','clarification_test')
    assert result['status']=='waiting_for_clarification'
    assert result['report']['steps']==2 and result['report']['question']=='Which source?'
    client=TestClient(app)
    data=client.get('/api/runs/clarification_test').json()
    assert data['working_memory'] and data['report']['status']=='waiting_for_clarification'
    observation=next(event for event in data['events'] if event['event_type']=='OBSERVATION')
    assert observation['payload']['error_code']=='AMBIGUOUS' and observation['payload']['evidence']=={'source_id':'one.txt'}
    stream=client.get('/api/runs/clarification_test/stream')
    assert stream.status_code==200 and 'STREAM_END' in stream.text
    last=data['events'][-1]['id']
    replay=client.get('/api/runs/clarification_test/stream',headers={'Last-Event-ID':str(last)})
    assert 'CLARIFICATION_NEEDED' not in replay.text and 'STREAM_END' in replay.text
    assert client.get('/api/runs/missing/stream').status_code==404

@pytest.mark.anyio
async def test_verifier_exception_ends_failed_releases_lease():
    class BrokenVerifier(VerifierEngine):
        async def verify_run(self,*args,**kwargs): raise RuntimeError('Unavailable')
    runner=AgentRunner(provider=provider(AgentDecision(thought='Check',action='ready_for_verification')),verifier=BrokenVerifier())
    result=await runner.execute_task('Check','broken_verifier')
    assert result['status']=='failed' and 'RuntimeError' in result['error']
    with get_db_connection() as conn:
        assert conn.execute('SELECT COUNT(*) FROM workspace_lease').fetchone()[0]==0
        assert conn.execute("SELECT working_memory FROM runs WHERE run_id='broken_verifier'").fetchone()[0]

@pytest.mark.anyio
async def test_deadline_and_step_budget_terminate():
    class SlowProvider(FakeProvider):
        async def generate_structured(self,*args,**kwargs):
            await asyncio.sleep(1)
    runner=AgentRunner(provider=SlowProvider(),run_deadline_seconds=.01)
    result=await runner.execute_task('Check','deadline_test')
    assert result['status']=='failed' and result['error']=='Run deadline exceeded'
    with get_db_connection() as conn:
        assert conn.execute('SELECT COUNT(*) FROM workspace_lease').fetchone()[0]==0
    registry=ToolRegistry(); registry.register('inspect','Inspect',{},lambda:ToolResult(ok=False,error='Still failing'))
    runner=AgentRunner(provider=provider(AgentDecision(thought='Inspect',action='act',tool_name='inspect',tool_args={})),tool_registry=registry,max_steps=1)
    result=await runner.execute_task('Check','bounded_test')
    assert result['status']=='failed' and 'step budget' in result['error']

@pytest.mark.parametrize('payload',[{'objective':' '},{'objective':''},{'objective':'Check','max_steps':999},{'objective':'Check','max_steps':0},{'objective':'Check','run_id':'../another'}])
def test_invalid_run_request(payload):
    assert TestClient(app).post('/api/runs',json=payload).status_code==422

@pytest.mark.anyio
async def test_failed_verification_keeps_source_and_actual_expected_in_report():
    runner=AgentRunner(provider=provider(AgentDecision(thought='Check',action='ready_for_verification')),verifier=VerifierEngine(intent=VerificationIntent(collection='invoices',company='Acme Corp',selection='latest')),max_verification_attempts=1)
    result=await runner.execute_task('Enter latest Acme Corp invoice','missing_record')
    assert result['status']=='failed'
    checks=result['report']['verification']['criteria_results']
    assert any(check['evidence'].get('source_id')=='acme_invoice_1044.pdf' for check in checks)
    assert result['report']['verification']['discrepancies']
    assert result['report']['state_delta']['invoices']['created']==[]
