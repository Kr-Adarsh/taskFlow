import json
import pytest
from backend.app.agent.provider import FakeProvider
from backend.app.agent_v2.graph import AgentRunner
from backend.app.agent.schemas import VerificationIntent
from backend.app.agent_v2.state import TaskPlan, Decision
from backend.app.agent.verifier import snapshot_state
from backend.app.workspace.seed import reset_demo_env
from backend.app.workspace.db import get_db_connection

pytestmark=pytest.mark.anyio


def account_task(task_id,company,dependencies=()):
    return {'task_id':task_id,'goal':f'Read {company} account tier','dependencies':list(dependencies),'success_criteria':['Exact tier returned'],'verification_capability':'browser'}


async def test_single_task_verifies_original_objective_without_coverage_call():
    reset_demo_env()
    objective = 'Read Acme Corp customer_name, tier, mrr, account_manager and status'
    fields = ['customer_name', 'tier', 'mrr', 'account_manager', 'status']
    account = next(row for row in snapshot_state()['accounts'] if row['customer_name'] == 'Acme Corp')
    answer = {field: account[field] for field in fields}
    responses = [
        {'objective': objective, 'success_criteria': [objective], 'tasks': [account_task('account', 'Acme Corp')]},
        {'thought': 'Return account fields', 'action': 'ready_for_verification', 'result': answer},
        {'collection': 'accounts', 'company': 'Acme Corp', 'requested_fields': fields, 'require_new_record': False},
    ]

    class SingleTaskProvider(FakeProvider):
        expected_schemas = [TaskPlan, Decision, VerificationIntent]

        async def generate_structured(self, messages, response_schema, temperature=0.0):
            assert self.expected_schemas, 'Unexpected extra verifier model call'
            assert response_schema is self.expected_schemas.pop(0)
            return await super().generate_structured(messages, response_schema, temperature)

    provider = SingleTaskProvider(responses)
    result = await AgentRunner(provider=provider).execute_task(objective)
    assert result['status'] == 'completed', result
    assert not provider.responses and not provider.expected_schemas
    assert len(provider.call_history) == 3
    assert objective in provider.call_history[2][1]['content']
    assert result['verification']['verified']
    assert result['verification']['context']['original_objective'] == objective
    assert not any('CoverageAssessment' in str(messages) for messages in provider.call_history)


@pytest.mark.parametrize('coverage',[True,False])
async def test_dependency_results_and_original_coverage_control_completion(coverage):
    reset_demo_env()
    objective='Read Acme Corp and Globex Inc account tiers'
    plan={'objective':objective,'success_criteria':[objective],'tasks':[account_task('a','Acme Corp'),account_task('b','Globex Inc',['a'])]}
    provider=FakeProvider([plan,{'thought':'Answer','action':'ready_for_verification','result':{'customer_name':'Acme Corp','tier':'Enterprise'}},{'collection':'accounts','company':'Acme Corp','requested_fields':['tier'],'require_new_record':False},{'thought':'Answer second','action':'ready_for_verification','result':{'customer_name':'Globex Inc','tier':'Growth'}},{'collection':'accounts','company':'Globex Inc','requested_fields':['tier'],'require_new_record':False},{'covered':coverage,'missing_requirements':[] if coverage else ['Original requirement omitted'],'reason':'Covered' if coverage else 'Incomplete coverage'}])
    result=await AgentRunner(provider=provider).execute_task(objective)
    assert result['status']==('completed' if coverage else 'failed')
    assert not provider.responses
    assert 'CoverageAssessment' in provider.call_history[-1][0]['content']
    assert len(result['report']['tasks'])==2
    assert all(task['status']=='COMPLETED' for task in result['report']['tasks'])
    assert 'Enterprise' in json.dumps(provider.call_history[3])


async def test_failed_parent_blocks_dependents_and_releases_lease():
    reset_demo_env()
    plan={'objective':'Read accounts','success_criteria':['Both accounts checked'],'tasks':[account_task('a','Missing'),account_task('b','Globex Inc',['a'])]}
    provider=FakeProvider([plan,{'thought':'Missing account','action':'fail','failure_reason':'Account missing'}])
    result=await AgentRunner(provider=provider).execute_task('Read accounts')
    assert result['status']=='failed'
    assert [task['status'] for task in result['report']['tasks']]==['FAILED','BLOCKED']
    with get_db_connection() as conn:
        assert conn.execute('SELECT COUNT(*) FROM workspace_lease').fetchone()[0]==0


async def test_replan_is_bounded_and_keeps_step_budget():
    reset_demo_env()
    plan={'objective':'Read account','success_criteria':['Account checked'],'tasks':[account_task('a','Missing')]}
    replan={'thought':'Change plan','action':'ready_for_verification','replan':True}
    provider=FakeProvider([plan,replan,plan,replan])
    result=await AgentRunner(provider=provider).execute_task('Read account')
    assert result['status']=='failed' and 'Replan budget' in result['error']
    assert result['steps']==2
