"""Real V2 gates: one unchanged runtime, actual model, browser and local computation."""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import pytest
from test_real_acceptance import real_workspace as real_workspace, INVOICE, COMPLAINT
from backend.app.agent.provider import get_default_provider
from backend.app.agent_v2.graph import AgentRunner
from backend.app.agent_v2.context import SYSTEM_PROMPT
from backend.app.agent.verifier import snapshot_state, state_delta
from backend.app.workspace.fault_injection import fault_manager

pytestmark=[pytest.mark.anyio,pytest.mark.skipif(os.getenv('TASKFLOW_RUN_REAL')!='1',reason='Explicit real-model V2 gate')]

async def run_v2(case,objective):
    before=snapshot_state();events=[]
    build_hash=hashlib.sha256(b''.join(str(p).encode()+p.read_bytes() for p in sorted(Path('backend/app').rglob('*')) if p.suffix in ('.py','.html'))).hexdigest()
    provider=get_default_provider()
    original=provider.generate_structured
    generations=[]
    async def measured(messages,schema,*args,**kwargs):
        entry={'schema':schema.__name__};generations.append(entry)
        try:
            result,meta=await original(messages,schema,*args,**kwargs)
        except Exception as error:
            entry['error']=str(error)
            raise
        entry.update(response=result.model_dump(),metadata=meta)
        return result,meta
    provider.generate_structured=measured
    def track(run_id,kind,payload):
        events.append({'event_type':kind,'payload':payload})
        if kind=='ACTION':print(case,payload['step'],payload['tool'],flush=True)
        if kind in ('FAILURE','CLARIFICATION_NEEDED','VERIFICATION'):print(case,kind,payload.get('verified',payload.get('summary')),flush=True)
    start=time.perf_counter()
    interruption=None
    run_id='v2_'+uuid.uuid4().hex
    try:
        result=await AgentRunner(provider=provider,event_callback=track).execute_task(objective,run_id=run_id)
    except BaseException as error:
        interruption=error
        report=next((e['payload'] for e in reversed(events) if e['event_type']=='FINAL_REPORT'),{})
        result={'run_id':run_id,'status':report.get('status','interrupted'),'report':report}
    after=snapshot_state();delta=state_delta(before,after)
    record={'case':case,'objective':objective,'model':provider.model,'provider':type(provider).__name__,'provider_usage':provider.usage_snapshot(),'result':result,'pre_state':before,'post_state':after,'state_delta':delta,'events':events,'generations':generations,'tokens_available':sum(g.get('metadata',{}).get('prompt_eval_count',0)+g.get('metadata',{}).get('eval_count',0) for g in generations),'duration_seconds':round(time.perf_counter()-start,3),'system_prompt_sha256':hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),'build_sha256':build_hash}
    path=Path('docs/validation/v2-acceptance-results.json');path.parent.mkdir(exist_ok=True)
    records=json.loads(path.read_text()) if path.exists() else [];records.append(record)
    path.write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n')
    print(case,'FINAL',result['status'],'tokens',record['tokens_available'],'seconds',record['duration_seconds'],flush=True)
    if interruption is not None: raise interruption
    return result,delta,events

async def test_v2_invoice(real_workspace):
    fault_manager.arm('finance_create_invoice',count=1)
    result,delta,events=await run_v2('V2-A invoice',INVOICE)
    assert result['status']=='completed',result.get('error') or result.get('question')
    assert result['verification']['verified']
    rows=delta['invoices']['created'];assert len(rows)==1
    row=rows[0]
    assert (row['company'],row['invoice_number'],row['currency'],row['amount_minor'],row['due_date'],row['source_reference'])==('Acme Corp','INV-1044','INR',8450000,'2026-10-15','acme_invoice_1044.pdf')
    assert any((e['payload'].get('evidence') or {}).get('http_status')==503 for e in events if e['event_type']=='OBSERVATION')
    assert not any(delta[c][kind] for c in delta for kind in ('updated','deleted'))
    assert not delta['tickets']['created'] and not delta['accounts']['created']

async def test_v2_complaint(real_workspace):
    result,delta,_=await run_v2('V2-B complaint',COMPLAINT)
    assert result['status']=='completed',result.get('error') or result.get('question')
    assert result['verification']['verified'] and len(delta['tickets']['created'])==1
    row=delta['tickets']['created'][0]
    assert (row['customer'],row['priority'],row['source_reference'])==('Acme Corp','High','complaint_4821.txt')
    assert not any(delta[c][kind] for c in delta for kind in ('updated','deleted'))
    assert not delta['invoices']['created'] and not delta['accounts']['created']


DATASET = 'Analyze sales.csv and identify the region with the largest absolute revenue decline between 2026-08 and 2026-09, including the calculated decline.'

async def test_v2_dataset(real_workspace):
    result, delta, events = await run_v2('V2-C dataset', DATASET)
    assert result['status'] == 'completed', result.get('error') or result.get('question')
    assert result['verification']['verified']
    assert not any(rows for changes in delta.values() for rows in changes.values())
    executions = [event['payload']['data'] for event in events if event['event_type'] == 'OBSERVATION' and event['payload'].get('ok') and (event['payload'].get('data') or {}).get('stage') == 'full']
    assert executions and executions[-1]['inputs'][0]['rows'] == 800
    expected = result['verification']['evidence']['expected']
    assert expected['group'] == 'South' and expected['value'] == 27000
    assert any(event['payload'].get('tool') == 'profile_dataset' for event in events if event['event_type'] == 'ACTION')
