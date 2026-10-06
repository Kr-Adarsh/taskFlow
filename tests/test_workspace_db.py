"""
Focused tests for workspace database, seed/reset, domain validation,
duplicate protection, and deterministic fault injection.
"""

from pathlib import Path
import pytest
import sqlite3

from backend.app.workspace.db import init_db, get_db_connection
from backend.app.workspace.models import (
    InvoiceCreate,
    SupportTicketCreate,
    parse_money_to_minor,
    format_minor_to_money,
)
from backend.app.workspace.seed import seed_workspace, reset_demo_env
from backend.app.workspace.fault_injection import fault_manager, FaultInjectionError
from backend.app.workspace.service import (
    create_invoice,
    list_invoices,
    get_invoice,
    list_crm_accounts,
    get_crm_account_by_name,
    search_crm_accounts,
    create_support_ticket,
    list_support_tickets,
    list_documents,
    extract_document_text,
)

@pytest.fixture
def temp_db(tmp_path: Path):
    db_file = tmp_path / "test_workspace.db"
    seed_workspace(db_path=db_file, force_reseed=True)
    yield db_file

def test_money_conversion():
    assert parse_money_to_minor("84,500") == 8450000
    assert parse_money_to_minor("84500.00") == 8450000
    assert parse_money_to_minor(84500) == 8450000
    assert parse_money_to_minor("₹84,500") == 8450000
    assert format_minor_to_money(8450000, "INR") == "₹84,500.00"

def test_seed_reproducibility(temp_db: Path):
    invoices = list_invoices(db_path=temp_db)
    crm = list_crm_accounts(db_path=temp_db)
    tickets = list_support_tickets(db_path=temp_db)
    docs = list_documents(db_path=temp_db)

    assert len(invoices) == 2
    assert len(crm) == 3
    assert len(tickets) == 1
    assert len(docs) >= 6

    acme_crm = get_crm_account_by_name("Acme Corp", db_path=temp_db)
    assert acme_crm is not None
    assert acme_crm.tier == "Enterprise"

    beta_crm = get_crm_account_by_name("Beta Retail", db_path=temp_db)
    assert beta_crm is not None
    assert beta_crm.tier == "Starter"

def test_finance_invoice_create_and_duplicate_rejection(temp_db: Path):
    inv_data = InvoiceCreate(
        company="Acme Corp",
        invoice_number="INV-1044",
        amount="84,500",
        currency="INR",
        due_date="2026-10-15",
        status="Pending",
        source_reference="acme_invoice_1044.pdf"
    )
    created = create_invoice(inv_data, db_path=temp_db)
    assert created.id is not None
    assert created.amount_minor == 8450000
    assert created.amount_formatted == "₹84,500.00"
    assert created.invoice_number == "INV-1044"

    # Verify duplicate creation raises ValueError
    with pytest.raises(ValueError, match="Duplicate invoice"):
        create_invoice(inv_data, db_path=temp_db)

def test_support_ticket_create_and_list(temp_db: Path):
    ticket_data = SupportTicketCreate(
        customer="Acme Corp",
        priority="High",
        status="Open",
        source_reference="complaint_4821",
        summary="Critical EU database synchronization outage"
    )
    created = create_support_ticket(ticket_data, db_path=temp_db)
    assert created.ticket_id.startswith("TIK-")
    assert created.priority == "High"
    assert created.customer == "Acme Corp"
    assert "synchronization" in created.summary

    tickets = list_support_tickets(customer="Acme Corp", db_path=temp_db)
    assert len(tickets) == 1
    assert tickets[0].ticket_id == created.ticket_id

def test_document_extraction():
    fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures" / "documents"
    pdf_text = extract_document_text(fixtures_dir / "acme_invoice_1044.pdf")
    assert "INV-1044" in pdf_text
    assert "84,500" in pdf_text

    txt_text = extract_document_text(fixtures_dir / "complaint_4821.txt")
    assert "Acme Corp" in txt_text
    assert "4821" in txt_text

def test_fault_injection_and_recovery(temp_db: Path):
    fault_manager.reset()
    fault_manager.arm("finance_create_invoice", count=1)

    inv_data = InvoiceCreate(
        company="Acme Corp",
        invoice_number="INV-9999",
        amount="15,000",
        due_date="2026-11-01"
    )

    # 1. First attempt MUST raise FaultInjectionError
    with pytest.raises(FaultInjectionError) as exc_info:
        create_invoice(inv_data, db_path=temp_db)
    assert exc_info.value.retriable is True
    assert exc_info.value.error_code == "FINANCE_SERVICE_UNAVAILABLE"

    # Confirm NO record was committed
    invoices_after_fail = [i for i in list_invoices(db_path=temp_db) if i.invoice_number == "INV-9999"]
    assert len(invoices_after_fail) == 0

    # 2. Second attempt (retry) MUST succeed
    created = create_invoice(inv_data, db_path=temp_db)
    assert created.invoice_number == "INV-9999"
    assert created.amount_minor == 1500000

    # Confirm exactly ONE record now exists
    invoices_after_retry = [i for i in list_invoices(db_path=temp_db) if i.invoice_number == "INV-9999"]
    assert len(invoices_after_retry) == 1

def test_reset_demo_env(temp_db: Path):
    # Mutate state
    inv_data = InvoiceCreate(
        company="Test Corp",
        invoice_number="TEST-001",
        amount="1,000",
        due_date="2026-12-01"
    )
    create_invoice(inv_data, db_path=temp_db)
    assert len(list_invoices(db_path=temp_db)) == 3

    # Reset
    reset_demo_env(db_path=temp_db)

    # Invoices back to baseline of 2
    assert len(list_invoices(db_path=temp_db)) == 2
    test_invs = [i for i in list_invoices(db_path=temp_db) if i.company == "Test Corp"]
    assert len(test_invs) == 0
