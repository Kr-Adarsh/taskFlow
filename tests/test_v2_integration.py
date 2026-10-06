"""Scripted decisions, real graph/browser/database/sandbox. Not autonomy evidence."""
import json
import pytest
from fastapi.testclient import TestClient
from test_real_acceptance import real_workspace as real_workspace, INVOICE, COMPLAINT
from backend.app.main import app
from backend.app.agent.provider import FakeProvider
from backend.app.agent_v2.graph import AgentRunner
from backend.app.agent.verifier import snapshot_state, state_delta
from backend.app.workspace.db import get_db_connection
from backend.app.workspace.fault_injection import fault_manager

pytestmark=pytest.mark.anyio


def plan(objective, capability):
    return {'objective':objective,'success_criteria':[objective],'tasks':[{'task_id':'outcome','goal':objective,'success_criteria':[objective],'verification_capability':capability}]}


def act(name,**args):
    return {'thought':'Exercise observed operation','action':'act','tool_name':name,'tool_args':args}


async def test_invoice_v2_actual_browser_recovery(real_workspace):
    fault_manager.arm('finance_create_invoice',1)
    before=snapshot_state();events=[]
    steps=[plan(INVOICE,'browser'),act('read_document_chunks',document_id='acme_invoice_1044.pdf'),act('browser_open',url='/workspace/finance')]
    for field,text in [('company','Acme Corp'),('invoice_number','INV-1044'),('amount','84500'),('due_date','2026-10-15'),('source_reference','acme_invoice_1044.pdf')]:
        steps.append(act('browser_type',element_id='@'+field,text=text))
    steps.extend([act('browser_click',element_id='@submit_invoice'),act('browser_click',element_id='@submit_invoice'),{'collection':'invoices','company':'Acme Corp','selection':'latest'}])
    provider=FakeProvider(steps)
    result=await AgentRunner(provider=provider,event_callback=lambda run_id,kind,payload:events.append((kind,payload))).execute_task(INVOICE)
    assert result['status']=='completed',result
    assert not provider.responses
    delta=state_delta(before,snapshot_state())
    assert len(delta['invoices']['created'])==1 and delta['invoices']['created'][0]['amount_minor']==8450000
    assert not any(delta[name][change] for name in delta for change in ('updated','deleted'))
    assert any(kind=='OBSERVATION' and payload.get('evidence',{}).get('http_status')==503 for kind,payload in events)
    assert result['report']['tasks'][0]['status']=='COMPLETED'
    with get_db_connection() as conn:
        persisted=json.loads(conn.execute("SELECT payload FROM run_events WHERE run_id=? AND event_type='FINAL_REPORT'",(result['run_id'],)).fetchone()[0])
    assert persisted['runtime']=='v2' and persisted['tasks'][0]['result']['verification']['verified']


@pytest.mark.parametrize('enterprise',[True,False])
async def test_complaint_same_graph_real_browser_and_conditional_no_op(real_workspace,enterprise):
    objective=COMPLAINT if enterprise else COMPLAINT.replace('4821','4822')
    source='complaint_4821.txt' if enterprise else 'complaint_4822.txt'
    before=snapshot_state()
    steps=[plan(objective,'browser'),act('read_document_chunks',document_id=source),act('browser_open',url='/workspace/crm')]
    if enterprise:
        steps.append(act('browser_open',url='/workspace/support'))
        steps.extend([act('browser_type',element_id='@ticket_customer',text='Acme Corp'),act('browser_select',element_id='@ticket_priority',value='High'),act('browser_type',element_id='@ticket_source_ref',text=source),act('browser_type',element_id='@ticket_summary',text='Severe database synchronization outages across EU servers; customer operations are affected.'),act('browser_click',element_id='@submit_ticket')])
    if not enterprise:
        steps.append({'thought':'Request independent check','action':'ready_for_verification'})
    steps.append({'collection':'tickets','complaint_id':'4821' if enterprise else '4822','condition_tier':'Enterprise','priority':'High'})
    if enterprise:
        steps.append({'accurate':True,'reason':'Matches source','source_quotes':['severe database synchronization outages'],'contradictions':[]})
    provider=FakeProvider(steps)
    result=await AgentRunner(provider=provider).execute_task(objective)
    assert result['status']=='completed',result
    assert not provider.responses
    delta=state_delta(before,snapshot_state())
    assert len(delta['tickets']['created'])==(1 if enterprise else 0)
    assert not delta['invoices']['created'] and not delta['accounts']['created']
    assert not any(delta[name][change] for name in delta for change in ('updated','deleted'))


async def test_dataset_same_graph_full_sandbox_and_independent_verifier(real_workspace,monkeypatch):
    monkeypatch.setenv('TASKFLOW_ARTIFACTS_DIR',str(real_workspace/'artifacts'))
    objective='Identify the region with the largest absolute revenue decline in sales.csv from 2026-08 to 2026-09, including the decline.'
    code="""df=pd.read_csv(inputs['sales.csv'])
totals=df.groupby(['region','month'])['revenue'].sum().unstack()
decline=totals['2026-08']-totals['2026-09']
result={'summary':'Compared full-data revenue totals','metrics':{'region':str(decline.idxmax()),'decline':float(decline.max())},'tables':{'comparison':totals.reset_index()}}
"""
    steps=[plan(objective,'python'),act('profile_dataset',document_id='sales.csv'),act('execute_python',document_ids=['sales.csv'],code=code),{'thought':'Check computed values','action':'ready_for_verification','result':{'metrics':{'region':'South','decline':27000.0}}}, {'document_id':'sales.csv','group_column':'region','value_column':'revenue','aggregate':'sum','period_column':'month','baseline_period':'2026-08','current_period':'2026-09','measure':'difference','convention':'baseline_minus_current','selection':'max','group_metric':'region','value_metric':'decline'}]
    before=snapshot_state()
    provider=FakeProvider(steps)
    result=await AgentRunner(provider=provider).execute_task(objective)
    assert result['status']=='completed',result
    assert not provider.responses
    expected=result['verification']['evidence']['expected']
    assert expected['full_dataset_rows']==800 and expected['value']==27000
    assert not any(rows for changes in state_delta(before,snapshot_state()).values() for rows in changes.values())
    reference=result['report']['artifacts'][0]['reference']
    client=TestClient(app)
    download=client.get(reference)
    assert download.status_code==200 and 'South' in download.text
    assert client.get(reference.replace(result['run_id'],'different_run')).status_code==404
    profile=client.get('/api/workspace/documents/sales.csv/content').json()['profile']
    assert len(profile['sample_rows'])==3
    assert 'North,2026-08,1200' not in client.get('/workspace/documents?view=sales.csv').text


async def test_computation_cannot_complete_with_wrong_authoritative_metrics(real_workspace,monkeypatch):
    monkeypatch.setenv('TASKFLOW_ARTIFACTS_DIR',str(real_workspace/'artifacts'))
    objective='Identify the region with the largest revenue decline between 2026-08 and 2026-09'
    steps=[plan(objective,'python'),act('profile_dataset',document_id='sales.csv'),
        act('execute_python',document_ids=['sales.csv'],code="result={'region':'North','decline':1}"),
        {'thought':'Claim correct value without computing it','action':'ready_for_verification','result':{'metrics':{'region':'South','decline':27000}}},
        {'document_id':'sales.csv','group_column':'region','value_column':'revenue','aggregate':'sum',
         'period_column':'month','baseline_period':'2026-08','current_period':'2026-09','measure':'difference',
         'convention':'baseline_minus_current','selection':'max','group_metric':'region','value_metric':'decline'},
        {'thought':'Acknowledge independent rejection','action':'fail','failure_reason':'Computed result was incorrect'}]
    provider=FakeProvider(steps)
    result=await AgentRunner(provider=provider).execute_task(objective)
    assert result['status']=='failed'
    assert not provider.responses
    task=result['report']['tasks'][0]
    assert task['result']['metrics']=={'region':'North','decline':1}
    assert task['model_reported_result']['metrics']=={'region':'South','decline':27000}
    assert task['model_result_matches_authoritative'] is False
    assert result['report']['verification']['verified'] is False
    assert 'independent' in result['report']['verification']['summary']
