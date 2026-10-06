"""Mutation ownership, idempotency, restart and screenshot boundaries."""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.workspace.db import get_db_connection
from backend.app.workspace.lease import reserve_run, release_run, WorkspaceBusy, recover_interrupted
from backend.app.workspace.seed import reset_demo_env
from backend.app.workspace.models import InvoiceCreate, SupportTicketCreate
from backend.app.workspace.service import create_invoice, create_support_ticket
from backend.app.tools.browser_tools import browser_manager

@pytest.fixture(autouse=True)
def reset():
    reset_demo_env()
    yield
    with get_db_connection() as conn:
        conn.execute('DELETE FROM workspace_lease')
        conn.commit()


def test_ticket_repeated_submission_idempotent_and_wrong_record_not_repaired():
    payload=SupportTicketCreate(customer='Acme Corp',priority='High',summary='Database cluster outage',source_reference='4821')
    first=create_support_ticket(payload)
    assert create_support_ticket(payload).id==first.id
    assert create_support_ticket(payload.model_copy(update={'source_reference':'complaint_4821'})).id==first.id
    with pytest.raises(ValueError,match='correction'):
        create_support_ticket(payload.model_copy(update={'priority':'Medium'}))
    with get_db_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM support_tickets WHERE customer='Acme Corp'").fetchone()[0]==1


@pytest.mark.parametrize('reference', ['Complaint #4821', 'complaint_9999', 'unregistered-source'])
def test_unresolved_ticket_reference_rejected_before_mutation(reference):
    client = TestClient(app)
    with get_db_connection() as conn:
        before = conn.execute('SELECT COUNT(*) FROM support_tickets').fetchone()[0]
    response = client.post('/api/workspace/tickets', json={
        'customer': 'Acme Corp', 'priority': 'High',
        'summary': 'Database cluster outage', 'source_reference': reference,
    })
    assert response.status_code == 400
    assert 'exact filename' in response.text
    with get_db_connection() as conn:
        assert conn.execute('SELECT COUNT(*) FROM support_tickets').fetchone()[0] == before


def test_case_insensitive_invoice_replacement():
    payload=InvoiceCreate(company='Acme Corp',invoice_number='INV-1005',amount='100',due_date='2026-10-01')
    replaced = create_invoice(payload.model_copy(update={'company':' acme corp ', 'invoice_number':'inv-1005'}))
    assert replaced.company == 'Acme Corp' and replaced.invoice_number == 'INV-1005'
    assert replaced.amount_minor == 10000
    with get_db_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM finance_invoices WHERE LOWER(company)='acme corp'").fetchone()[0] == 1


def test_concurrent_reservations_and_manual_write_reset_blocked():
    def reserve(name):
        try:
            reserve_run('Objective',name)
            return True
        except WorkspaceBusy:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(reserve,['owner_one','owner_two']))==[False,True]
    client=TestClient(app)
    for endpoint,body in [('/api/workspace/reset',{}),('/api/workspace/fault-injection',{}),('/api/workspace/tickets',{'customer':'A','summary':'Issue','source_reference':'s'})]:
        assert client.post(endpoint,json=body).status_code==409
    with get_db_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM runs WHERE run_id IN ('owner_one','owner_two')").fetchone()[0]==1
        owner=conn.execute('SELECT run_id FROM workspace_lease').fetchone()[0]
    release_run(owner)
    with pytest.raises(WorkspaceBusy,match='already exists'):
        reserve_run('Changed objective',owner)
    with get_db_connection() as conn:
        assert conn.execute('SELECT objective FROM runs WHERE run_id=?',(owner,)).fetchone()[0]=='Objective'


def test_restart_marks_interrupted_releases_lease():
    reserve_run('Objective','interrupted_test')
    recover_interrupted()
    with get_db_connection() as conn:
        assert conn.execute("SELECT status FROM runs WHERE run_id='interrupted_test'").fetchone()[0]=='interrupted'
        assert conn.execute('SELECT COUNT(*) FROM workspace_lease').fetchone()[0]==0


@pytest.mark.parametrize('value',['junk84.5','-5','1.234','NaN','12,34','Infinity'])
def test_strict_money(value):
    with pytest.raises(ValueError):
        InvoiceCreate(company='A',invoice_number='B',amount=value,due_date='2026-10-01')


def test_screenshot_cannot_leak_between_runs(tmp_path):
    client=TestClient(app)
    reserve_run('One','screenshot_one'); release_run('screenshot_one')
    reserve_run('Two','screenshot_two'); release_run('screenshot_two')
    path=browser_manager.screenshot_root/'screenshot_one'/'latest.png'
    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b'own image')
    with get_db_connection() as conn:
        for name in ['screenshot_one','screenshot_two']:
            conn.execute("INSERT INTO run_events(run_id,event_type,payload,timestamp) VALUES(?,?,?,?)",(name,'OBSERVATION',json.dumps({'evidence':{'screenshot':str(path)}}),'now'))
        conn.commit()
    assert client.get('/api/runs/screenshot_one/screenshot').content==b'own image'
    assert client.get('/api/runs/screenshot_two/screenshot').status_code==404
    assert client.get('/api/runs/missing/screenshot').status_code==404
