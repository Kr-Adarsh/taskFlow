"""Opt-in acceptance: real Groq, executor, browser, HTTP forms and isolated SQLite."""
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time
import uuid

import httpx
import pytest
import uvicorn

from backend.app.main import app
from backend.app.agent.provider import get_default_provider
from backend.app.agent.loop import AgentRunner
from backend.app.agent.verifier import snapshot_state, state_delta, source_invoice, VerifierEngine
from backend.app.agent.prompts import SYSTEM_PROMPT
from backend.app.tools.browser_tools import browser_manager
from backend.app.workspace.seed import reset_demo_env
from backend.app.workspace.fault_injection import fault_manager
from backend.app.workspace.db import get_db_connection
from backend.app.workspace.models import InvoiceCreate
from backend.app.workspace.service import create_invoice, list_documents, extract_document_text

pytestmark = [pytest.mark.anyio, pytest.mark.skipif(os.getenv('OPERON_RUN_REAL') != '1', reason='Explicit real-model API opt-in required')]
INVOICE = 'Find the latest invoice from Acme Corp, extract the amount and due date, enter it into Finance, and verify that it was recorded correctly.'
COMPLAINT = 'Read complaint 4821, identify the customer, check their CRM account, and if they are an Enterprise customer, create a high-priority support ticket summarizing the complaint.'
ARTIFACT = Path('docs/validation/acceptance-results.json')

@pytest.fixture
def real_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv('OPERON_DB_PATH',str(tmp_path/'workspace.db'))
    monkeypatch.setenv('OPERON_SCREENSHOTS_DIR',str(tmp_path/'screenshots'))
    monkeypatch.setenv('OPERON_ARTIFACTS_DIR',str(tmp_path/'artifacts'))
    reset_demo_env()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    previous=browser_manager.base_url
    browser_manager.base_url=f'http://127.0.0.1:{port}'
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='warning'))
    thread=threading.Thread(target=server.run,daemon=True); thread.start()
    try:
        for _ in range(100):
            if server.started: break
            time.sleep(.02)
        assert server.started, 'Isolated workspace server did not start'
        yield tmp_path
    finally:
        server.should_exit=True; thread.join(timeout=5)
        browser_manager.base_url=previous
        fault_manager.reset()
        assert not thread.is_alive(), 'Workspace server did not stop'

async def run_case(name,objective):
    before=snapshot_state()
    events=[]
    provider=get_default_provider()
    def track(run_id,kind,payload):
        events.append({'event_type':kind,'payload':payload})
        if kind=='ACTION': print(f"{name} step {payload['step']}: {payload['tool']}",flush=True)
        elif kind=='OBSERVATION' and not payload['ok']: print(f"{name} tool error: {payload.get('error_code')}",flush=True)
        elif kind in ('VERIFICATION','FAILURE','CLARIFICATION_NEEDED'): print(f"{name}: {kind} {payload.get('verified', payload.get('error', payload.get('question')))}",flush=True)
    runner=AgentRunner(provider=provider,verifier=VerifierEngine(provider=provider),event_callback=track)
    started=time.monotonic()
    run_id='acceptance_'+uuid.uuid4().hex
    interruption=None
    try:
        result=await runner.execute_task(objective,run_id=run_id)
    except BaseException as error:
        interruption=error
        report=next((event['payload'] for event in reversed(events) if event['event_type']=='FINAL_REPORT'),{})
        result={'run_id':run_id,'status':report.get('status','interrupted'),'steps':report.get('steps',0),'report':report}
    delta=state_delta(before,snapshot_state())
    verification=result.get('report',{}).get('verification')
    errors=[event['payload'] for event in events if event['event_type'] in ('ERROR','FAILURE') or (event['event_type']=='OBSERVATION' and not event['payload']['ok'])]
    unwanted=[check['evidence'].get('unwanted_mutations',[]) for check in (verification or {}).get('criteria_results',[]) if check['criterion']=='No unwanted mutations']
    summary={'case':name,'run_id':result['run_id'],'objective':objective,'status':result['status'],'steps':result.get('steps'),'duration_seconds':round(time.monotonic()-started,3),'provider':type(provider).__name__,'model':provider.model,'errors':errors,'state_delta':delta,'verification':verification,'unwanted_mutations':unwanted,'clarification_or_failure':result['status']!='completed','system_prompt_sha256':hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),'runtime_sha256':hashlib.sha256(b''.join(str(p).encode()+p.read_bytes() for p in sorted(Path('backend/app').rglob('*')) if p.suffix in ('.py','.html'))).hexdigest(),'events':events}
    ARTIFACT.parent.mkdir(parents=True,exist_ok=True)
    previous=json.loads(ARTIFACT.read_text()) if ARTIFACT.exists() else []
    previous.append(summary)
    ARTIFACT.write_text(json.dumps(previous,indent=2,ensure_ascii=False)+'\n')
    print(f"{name}: status={result['status']} steps={result.get('steps')} duration={summary['duration_seconds']} errors={len(errors)}",flush=True)
    if interruption is not None:
        raise interruption
    return result,delta,events


def no_other_mutations(delta, collection):
    assert all(not delta[name]['updated'] and not delta[name]['deleted'] for name in delta)
    assert all(not delta[name]['created'] for name in delta if name!=collection)

async def test_invoice(real_workspace):
    fault_manager.arm('finance_create_invoice',count=1)
    result,delta,events=await run_case('Invoice',INVOICE)
    assert result['status']=='completed', result.get('error') or result.get('question')
    assert result['verification']['verified']
    assert len(delta['invoices']['created'])==1
    row=delta['invoices']['created'][0]
    assert {key:row[key] for key in ['company','invoice_number','currency','amount_minor','due_date','source_reference']}=={'company':'Acme Corp','invoice_number':'INV-1044','currency':'INR','amount_minor':8450000,'due_date':'2026-10-15','source_reference':'acme_invoice_1044.pdf'}
    no_other_mutations(delta,'invoices')
    errors=[event for event in events if event['event_type']=='OBSERVATION' and not event['payload']['ok']]
    assert any(event['payload']['evidence'].get('http_status')==503 for event in errors)

async def test_complaint(real_workspace):
    result,delta,events=await run_case('Complaint',COMPLAINT)
    assert result['status']=='completed',result.get('error') or result.get('question')
    assert result['verification']['verified'] and len(delta['tickets']['created'])==1
    row=delta['tickets']['created'][0]
    assert row['customer']=='Acme Corp' and row['priority']=='High' and row['source_reference']=='complaint_4821.txt'
    no_other_mutations(delta,'tickets')

async def test_non_enterprise(real_workspace):
    result,delta,_=await run_case('Non-Enterprise',COMPLAINT.replace('4821','4822'))
    assert result['status']=='completed',result.get('error') or result.get('question')
    assert result['verification']['verified']
    assert all(not changes for collection in delta.values() for changes in collection.values())
    assert any(check['evidence'].get('conditional_outcome')=='no_op' for check in result['verification']['criteria_results'])

async def test_held_out(real_workspace):
    token=uuid.uuid4().hex[:6]
    company=f'Cedar Harbor {token}'
    number='CH-'+token.upper()
    filename=f'cedar_{token}.txt'
    text=f'Vendor: {company}\nInvoice Number: {number}\nInvoice Date: 2026-10-04\nDue Date: 2026-11-12\nSubtotal: 212.50 USD\nTax: 17.29 USD\nTotal Amount: 229.79 USD\nCurrency: USD\n'
    path=real_workspace/filename; path.write_text(text)
    with get_db_connection() as conn:
        conn.execute('INSERT INTO documents_index(filename,filepath,title,doc_type,company,doc_date,created_at) VALUES(?,?,?,?,?,?,?)',(filename,str(path),'New supplier invoice','invoice',company,'2026-10-04','now')); conn.commit()
    objective=f'Find the latest invoice from {company}, extract the amount and due date, enter it into Finance, and verify that it was recorded correctly.'
    result,delta,_=await run_case('Held-out',objective)
    assert result['status']=='completed',result.get('error') or result.get('question')
    assert len(delta['invoices']['created'])==1 and result['verification']['verified']
    row=delta['invoices']['created'][0]
    assert row['company']==company and row['invoice_number']==number and row['amount_minor']==22979 and row['currency']=='USD' and row['source_reference']==filename
    no_other_mutations(delta,'invoices')

async def test_persistent_failure(real_workspace):
    fault_manager.arm('finance_create_invoice',count=100)
    result,delta,events=await run_case('Persistent failure',INVOICE)
    assert result['status'] in ('failed','waiting_for_clarification')
    assert all(not changes for collection in delta.values() for changes in collection.values())
    assert any(event['event_type']=='OBSERVATION' and (event['payload'].get('evidence') or {}).get('http_status')==503 for event in events), 'Persistent Finance failure was not actually exercised'
    assert not any('Groq HTTP' in str(event['payload'].get('error','')) or 'generation failed' in str(event['payload'].get('error','')) for event in events), 'Provider failure does not prove business-failure recovery'
    assert not any(event['event_type']=='COMPLETE' for event in events)

async def test_wrong_record_repair_limitation(real_workspace):
    create_invoice(InvoiceCreate(company='Acme Corp',invoice_number='INV-1044',amount='1.00',currency='INR',due_date='2026-10-15',source_reference='acme_invoice_1044.pdf'))
    result,delta,events=await run_case('Repair limitation',INVOICE)
    assert not any('Groq HTTP' in str(event['payload'].get('error','')) or 'generation failed' in str(event['payload'].get('error','')) for event in events)
    assert any(event['event_type']=='OBSERVATION' and '/workspace/finance' in str(event['payload'].get('data')) for event in events), 'Wrong persisted state was not actually observed'
    assert result['status'] in ('failed','waiting_for_clarification')
    assert all(not changes for collection in delta.values() for changes in collection.values())
    assert not any(event['event_type']=='COMPLETE' for event in events)
