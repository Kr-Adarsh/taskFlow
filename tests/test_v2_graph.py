import pytest
from pydantic import ValidationError
from backend.app.agent.provider import FakeProvider
from backend.app.agent_v2.graph import AgentRunner
from backend.app.agent_v2.state import TaskPlan
from backend.app.workspace.seed import reset_demo_env
from backend.app.workspace.db import get_db_connection
from backend.app.capabilities.registry import build_registry


def plan(**changes):
    value={'objective':'Check source','success_criteria':['Source checked'],'tasks':[{'task_id':'source','goal':'Check source','dependencies':[], 'success_criteria':['Source checked'],'verification_capability':'documents'}]}
    value.update(changes)
    return value


@pytest.mark.parametrize('tasks',[
    [{'task_id':'a','goal':'A','dependencies':['b'],'success_criteria':['Checked'],'verification_capability':'documents'}],
    [{'task_id':key,'goal':key,'dependencies':[dep],'success_criteria':['Checked'],'verification_capability':'documents'} for key,dep in [('a','b'),('b','a')]],
])
def test_dependency_errors_rejected(tasks):
    with pytest.raises(ValidationError):TaskPlan.model_validate(plan(tasks=tasks))


@pytest.mark.anyio
async def test_graph_clarification_is_persisted_and_lease_released():
    reset_demo_env()
    provider=FakeProvider([plan(),{'thought':'Missing source','action':'need_clarification','clarification_question':'Which source?'}])
    result=await AgentRunner(provider=provider).execute_task('Check source')
    assert result['status']=='waiting_for_clarification'
    with get_db_connection() as conn:
        assert conn.execute('SELECT COUNT(*) FROM workspace_lease').fetchone()[0]==0
        assert conn.execute("SELECT COUNT(*) FROM run_events WHERE run_id=? AND event_type='TASK_GRAPH'",(result['run_id'],)).fetchone()[0]>=1


@pytest.mark.anyio
async def test_ungrounded_document_completion_cannot_pass():
    reset_demo_env()
    provider=FakeProvider([plan(),{'thought':'Done','action':'ready_for_verification','result':{'answer':'Invented answer'}}])
    result=await AgentRunner(provider=provider).execute_task('Check source')
    assert result['status']!='completed'


def test_registry_has_generic_capabilities_no_business_shortcuts():
    names=set(build_registry().categories)
    assert {'inspect_file','search_documents','read_document_chunks','browser_click'}<=names
    assert not any(word in name for name in names for word in ('invoice','complaint','enterprise'))
