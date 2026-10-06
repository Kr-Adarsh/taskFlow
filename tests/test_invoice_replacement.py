"""Repeat invoice intake keeps one record and still requires independent proof."""
from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest
from fastapi.testclient import TestClient

from backend.app.agent.provider import FakeProvider
from backend.app.agent.schemas import VerificationIntent
from backend.app.agent.verifier import VerifierEngine, snapshot_state, state_delta
from backend.app.agent_v2.graph import AgentRunner
from backend.app.main import app
from backend.app.workspace.db import init_db, get_db_connection
from backend.app.workspace.fault_injection import fault_manager, FaultInjectionError
from backend.app.workspace.models import InvoiceCreate
from backend.app.workspace.seed import seed_workspace
from backend.app.workspace.service import create_invoice
from test_real_acceptance import real_workspace, INVOICE
from test_v2_integration import act, plan


def payload(**changes):
    return InvoiceCreate(**{
        'company': 'Acme Corp', 'invoice_number': 'INV-1044', 'amount': '84500',
        'currency': 'INR', 'due_date': '2026-10-15', 'source_reference': 'acme_invoice_1044.pdf',
        **changes,
    })


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    path = tmp_path / 'replacement.db'
    monkeypatch.setenv('TASKFLOW_DB_PATH', str(path))
    seed_workspace(path, force_reseed=True)
    fault_manager.reset()
    yield path
    fault_manager.reset()


def test_same_submission_records_update_without_changing_identity(workspace):
    first = create_invoice(payload())
    before = snapshot_state()
    second = create_invoice(payload())
    assert second.id == first.id and second.created_at == first.created_at
    assert second.updated_at and second.updated_at != first.updated_at
    delta = state_delta(before, snapshot_state())
    assert not delta['invoices']['created'] and not delta['invoices']['deleted']
    row, = delta['invoices']['updated']
    assert row['id'] == first.id
    assert {key for key in row if row[key] != before['invoices'][-1].get(key)} == {'updated_at'}
    assert not any(rows for name in ('accounts', 'tickets') for rows in delta[name].values())


def test_replacement_updates_submitted_fields_preserves_payment_state(workspace):
    first = create_invoice(payload(status='Paid'))
    changed = create_invoice(payload(company=' acme corp ', invoice_number=' inv-1044 ',
                                     amount='123.45', currency='USD', due_date='2026-11-01',
                                     source_reference='replacement.pdf'))
    assert changed.id == first.id and changed.created_at == first.created_at
    assert (changed.company, changed.invoice_number) == ('Acme Corp', 'INV-1044')
    assert (changed.amount_minor, changed.currency, changed.due_date, changed.source_reference) == (
        12345, 'USD', '2026-11-01', 'replacement.pdf')
    assert changed.status == 'Paid'
    assert create_invoice(payload(status='Overdue')).status == 'Overdue'


def test_duplicate_api_submission_succeeds_with_same_id(workspace):
    client = TestClient(app)
    first = client.post('/api/workspace/invoices', json=payload().model_dump()).json()['data']
    second = client.post('/api/workspace/invoices', json=payload(amount='900').model_dump())
    assert second.status_code == 200
    assert second.json()['data']['id'] == first['id']
    assert second.json()['data']['amount_minor'] == 90000
    assert second.json()['data']['created_at'] == first['created_at']


def test_precommit_503_does_not_change_existing_invoice(workspace):
    first = create_invoice(payload())
    before = snapshot_state()
    fault_manager.arm('finance_create_invoice', 1)
    with pytest.raises(FaultInjectionError):
        create_invoice(payload(amount='123'))
    assert snapshot_state() == before
    assert create_invoice(payload(amount='123')).id == first.id


def test_concurrent_duplicate_submissions_keep_one_identity(workspace):
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda _: create_invoice(payload(), workspace), range(4)))
    assert len({row.id for row in rows}) == 1
    assert len({row.created_at for row in rows}) == 1
    with get_db_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM finance_invoices WHERE invoice_number='INV-1044'").fetchone()[0] == 1


def test_old_database_migration_preserves_invoice(tmp_path):
    path = tmp_path / 'old.db'
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE finance_invoices (id INTEGER PRIMARY KEY AUTOINCREMENT, '
                           'company TEXT, invoice_number TEXT, amount_minor INTEGER, currency TEXT, '
                           'due_date TEXT, status TEXT, created_at TEXT, source_reference TEXT)')
        connection.execute("INSERT INTO finance_invoices VALUES (1,'Vendor','REF-1',123,'USD',"
                           "'2026-10-01','Paid','original','ref.txt')")
    init_db(path)
    with get_db_connection(path) as connection:
        row = dict(connection.execute('SELECT * FROM finance_invoices').fetchone())
    assert row['updated_at'] is None and row['created_at'] == 'original'
    replacement = create_invoice(InvoiceCreate(company='Vendor', invoice_number='REF-1',
                                               amount='2', currency='USD', due_date='2026-10-02',
                                               source_reference='ref.txt'), path)
    assert replacement.id == 1 and replacement.created_at == 'original' and replacement.status == 'Paid'


@pytest.mark.anyio
@pytest.mark.parametrize('changes,passed', [({}, True), ({'amount': '1'}, False),
                                         ({'source_reference': 'unrelated.txt'}, False),
                                         ({'due_date': '2026-10-16'}, False)])
async def test_replacement_still_requires_source_matches(workspace, changes, passed):
    create_invoice(payload())
    verifier = VerifierEngine(workspace, intent=VerificationIntent(
        collection='invoices', company='Acme Corp', selection='latest', require_new_record=True))
    create_invoice(payload(**changes))
    result = await verifier.verify_run(INVOICE, {})
    assert result.verified is passed
    recorded = next(check for check in result.criteria_results
                    if check.criterion == 'Requested invoice recorded during this run')
    assert recorded.passed and recorded.evidence['replaced_during_run']


@pytest.mark.anyio
async def test_replacement_does_not_allow_unrelated_updates_or_deletes(workspace):
    create_invoice(payload())
    verifier = VerifierEngine(workspace, intent=VerificationIntent(
        collection='invoices', company='Acme Corp', selection='latest'))
    create_invoice(payload())
    with get_db_connection() as connection:
        connection.execute("UPDATE finance_invoices SET status='Overdue' WHERE invoice_number='INV-1005'")
        connection.execute("DELETE FROM support_tickets")
        connection.commit()
    result = await verifier.verify_run(INVOICE, {})
    unwanted = next(check for check in result.criteria_results if check.criterion == 'No unwanted mutations')
    assert not result.verified and not unwanted.passed
    assert {change['kind'] for change in unwanted.evidence['unwanted_mutations']} == {'updated', 'deleted'}


@pytest.mark.anyio
@pytest.mark.parametrize('incorrect_existing', [False, True])
async def test_duplicate_browser_submission_checkpoints_and_completes(real_workspace, incorrect_existing):
    existing = create_invoice(payload(amount='1' if incorrect_existing else '84500'))
    before, events = snapshot_state(), []
    fault_manager.arm('finance_create_invoice', 1)
    responses = [plan(INVOICE, 'browser'), act('read_document_chunks', document_id='acme_invoice_1044.pdf'),
                 act('browser_open', url='/workspace/finance')]
    for field, value in [('company', 'Acme Corp'), ('invoice_number', 'INV-1044'), ('amount', '84500'),
                         ('due_date', '2026-10-15'), ('source_reference', 'acme_invoice_1044.pdf')]:
        responses.append(act('browser_type', element_id='@' + field, text=value))
    responses.extend([act('browser_click', element_id='@submit_invoice'),
                      act('browser_click', element_id='@submit_invoice'),
                      {'collection': 'invoices', 'company': 'Acme Corp', 'selection': 'latest'}])
    provider = FakeProvider(responses)
    result = await AgentRunner(provider=provider, event_callback=lambda _, kind, data: events.append((kind, data))).execute_task(INVOICE)
    assert result['status'] == 'completed' and result['verification']['verified'] and not provider.responses
    failed = next(data for kind, data in events if kind == 'OBSERVATION' and data.get('error_code') == 'TRANSIENT_503_ERROR')
    assert failed['step'] == 8 and failed['evidence']['commit_state'] == 'not_committed'
    assert [data['step'] for kind, data in events if kind == 'VERIFICATION_CHECKPOINT'] == [9]
    assert [data['step'] for kind, data in events if kind == 'DECISION'] == list(range(1, 10))
    success = next(data for kind, data in events if kind == 'OBSERVATION' and data.get('data', {}).get('outcome') == 'confirmed_mutation')
    assert success['step'] == 9
    delta = state_delta(before, snapshot_state())
    row, = delta['invoices']['updated']
    assert row['id'] == existing.id and row['amount_minor'] == 8450000
    assert not delta['invoices']['created'] and not delta['invoices']['deleted']
    assert not any(rows for name in ('accounts', 'tickets') for rows in delta[name].values())
    final = next(data for kind, data in events if kind == 'VERIFICATION' and data.get('stage') == 'original_objective')
    assert final['verified']
