from datetime import datetime, timezone
from pathlib import Path
import pytest

from backend.app.agent.provider import FakeProvider
from backend.app.agent.schemas import VerificationIntent, SummaryAssessment
from backend.app.agent.verifier import VerifierEngine, snapshot_state, source_invoice
from backend.app.workspace.seed import seed_workspace
from backend.app.workspace.db import get_db_connection
from backend.app.workspace.service import create_invoice, create_support_ticket
from backend.app.workspace.models import InvoiceCreate, SupportTicketCreate


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "verify.db"
    seed_workspace(path, force_reseed=True)
    return path


def invoice_verifier(path, **kwargs):
    return VerifierEngine(path, intent=VerificationIntent(collection="invoices", company="Acme Corp", selection="latest", **kwargs))


def support_verifier(path, accurate=True, **kwargs):
    assessment = SummaryAssessment(accurate=accurate, reason="Grounded issue" if accurate else "Summary reverses or omits the reported outage", source_quotes=["severe database synchronization outages"] if accurate else [], contradictions=[] if accurate else ["No outage contradicts source"])
    return VerifierEngine(path, provider=FakeProvider([assessment]), intent=VerificationIntent(collection="tickets", complaint_id="4821", priority="High", condition_tier="Enterprise", **kwargs))


def invoice(path, **kwargs):
    values = dict(company="Acme Corp", invoice_number="INV-1044", amount="84500", due_date="2026-10-15", currency="INR", source_reference="acme_invoice_1044.pdf")
    values.update(kwargs)
    return create_invoice(InvoiceCreate(**values), path)


def ticket(path, **kwargs):
    values = dict(customer="Acme Corp", priority="High", source_reference="complaint_4821", summary="Severe database synchronization outages across EU servers and stalled analytics")
    values.update(kwargs)
    return create_support_ticket(SupportTicketCreate(**values), path)


async def verify(verifier, objective="Original request"):
    return await verifier.verify_run(objective, {}, success_criteria=["Requested record is source-linked and correct"])


@pytest.mark.anyio
async def test_correct_source_linked_invoice_delta(workspace):
    verifier = invoice_verifier(workspace)
    invoice(workspace)
    assert (await verify(verifier)).verified


@pytest.mark.anyio
@pytest.mark.parametrize("changes", [{"currency": "USD"}, {"company": "Acme"}, {"amount": "100"}, {"due_date": "2026-10-20"}, {"source_reference": "unrelated.pdf"}])
async def test_reject_wrong_invoice_fields(workspace, changes):
    verifier = invoice_verifier(workspace)
    invoice(workspace, **changes)
    assert not (await verify(verifier)).verified


@pytest.mark.anyio
async def test_specific_invoice_does_not_select_latest(workspace):
    verifier = VerifierEngine(workspace, intent=VerificationIntent(collection="invoices", company="Acme Corp", selection="specific", invoice_number="INV-1021"))
    invoice(workspace)
    assert not (await verify(verifier)).verified


@pytest.mark.anyio
async def test_preexisting_invoice_is_not_new_creation(workspace):
    invoice(workspace)
    verifier = invoice_verifier(workspace)
    assert not (await verify(verifier)).verified


@pytest.mark.anyio
async def test_no_ticket_wrong_reference_and_contradiction_rejected(workspace):
    verifier = support_verifier(workspace)
    ticket(workspace)
    with get_db_connection(workspace) as connection:
        connection.execute("UPDATE support_tickets SET source_reference='complaint_9999' WHERE customer='Acme Corp'")
        connection.commit()
    assert not (await verify(verifier)).verified
    with get_db_connection(workspace) as connection:
        connection.execute("UPDATE support_tickets SET source_reference='complaint_4821', summary='There are no outages; everything is working normally' WHERE customer='Acme Corp'")
        connection.commit()
    verifier.provider = FakeProvider([SummaryAssessment(accurate=False, reason="Contradictory summary", source_quotes=[], contradictions=["Source reports outage"])])
    assert not (await verify(verifier)).verified


@pytest.mark.anyio
async def test_correct_support_delta(workspace):
    verifier = support_verifier(workspace)
    ticket(workspace)
    assert (await verify(verifier)).verified


@pytest.mark.anyio
async def test_priority_is_taken_from_objective_contract(workspace):
    verifier = VerifierEngine(workspace, provider=FakeProvider([SummaryAssessment(accurate=True, reason="Grounded", source_quotes=["severe database synchronization outages"], contradictions=[])]), intent=VerificationIntent(collection="tickets", complaint_id="4821", priority="Medium", condition_tier="Enterprise"))
    ticket(workspace, priority="Medium")
    assert (await verify(verifier)).verified


@pytest.mark.anyio
async def test_duplicate_tickets_rejected(workspace):
    verifier = support_verifier(workspace)
    ticket(workspace)
    # Simulates imported/corrupt duplicates even after idempotency is implemented.
    with get_db_connection(workspace) as c:
        c.execute("INSERT INTO support_tickets(ticket_id,customer,priority,status,source_reference,summary,created_at) SELECT 'CORRUPT-1',customer,priority,status,source_reference,summary,created_at FROM support_tickets WHERE customer='Acme Corp'")
        c.commit()
    assert not (await verify(verifier)).verified


@pytest.mark.anyio
async def test_unknown_tier_unresolved(workspace):
    verifier = support_verifier(workspace)
    with get_db_connection(workspace) as c:
        c.execute("UPDATE crm_accounts SET tier='' WHERE customer_name='Acme Corp'")
        c.commit()
    assert not (await verify(verifier)).verified


@pytest.mark.anyio
async def test_nonenterprise_noop_allows_preexisting_tickets(workspace):
    ticket(workspace, customer="Beta Retail", source_reference="complaint_4822")
    verifier = VerifierEngine(workspace, intent=VerificationIntent(collection="tickets", complaint_id="4822", condition_tier="Enterprise"))
    assert (await verify(verifier)).verified
    ticket(workspace, customer="Beta Retail", source_reference="complaint_4821")
    assert not (await verify(verifier)).verified


@pytest.mark.anyio
async def test_old_ticket_cannot_satisfy_new_creation(workspace):
    ticket(workspace)
    verifier = support_verifier(workspace)
    assert not (await verify(verifier)).verified


@pytest.mark.parametrize("text", [
    "Vendor: Example\nInvoice Number: EX-1\nInvoice Date: 2026-01-01\nAmount: USD 100",
    "Vendor: Example\nInvoice Number: EX-1\nInvoice Date: 2026-01-01\nInvoice Date: 2026-02-01\nDue Date: 2026-03-01\nAmount: USD 100",
    "Vendor: Example\nInvoice Number: EX-1\nInvoice Date: 2026-01-01\nDue Date: 2026-03-01\nTotal: USD 100\nAmount Due: USD 200",
])
def test_missing_or_conflicting_source_values_rejected(text):
    with pytest.raises(ValueError):
        source_invoice(text)


def test_subtotal_is_not_payable_total():
    text = "Vendor: Example\nInvoice Number: EX-1\nInvoice Date: 2026-01-01\nDue Date: 2026-03-01\nSubtotal: USD 100\nTax: USD 20\nTotal: USD 120"
    assert source_invoice(text)["amount_minor"] == 12000


@pytest.mark.anyio
async def test_latest_tie_rejected(workspace, tmp_path):
    verifier = invoice_verifier(workspace)
    source = tmp_path / "other.txt"
    source.write_text("Vendor: Acme Corp\nInvoice Number: INV-9000\nInvoice Date: 2026-09-15\nDue Date: 2026-10-15\nAmount: INR 100")
    with get_db_connection(workspace) as c:
        c.execute("INSERT INTO documents_index(filename,filepath,title,doc_type,company,doc_date,content_preview,created_at) VALUES (?,?,?,'invoice','Acme Corp','2026-01-01','','now')", (source.name, str(source), "Other"))
        c.commit()
    invoice(workspace)
    assert not (await verify(verifier)).verified


@pytest.mark.anyio
async def test_original_objective_and_criteria_reach_independent_interpreter(workspace):
    provider = FakeProvider([VerificationIntent(collection="unsupported", unsupported_criteria=["Cannot update records"] )] * 2)
    verifier = VerifierEngine(workspace, provider=provider)
    assert not (await verifier.verify_run("Change existing account", {}, success_criteria=["Account changed"])).verified
    assert "Change existing account" in str(provider.call_history)
    assert "Account changed" in str(provider.call_history)


@pytest.mark.anyio
async def test_summary_source_quote_line_wrap_is_not_a_false_negative(workspace):
    verifier = support_verifier(workspace)
    ticket(workspace)
    quote = 'We are experiencing severe database synchronization outages across our EU servers affecting hundreds of our enterprise users.'
    verifier.provider = FakeProvider([SummaryAssessment(accurate=True, reason='Accurate summary', source_quotes=[quote], contradictions=[])])
    assert (await verify(verifier)).verified


@pytest.mark.anyio
async def test_summary_invalid_quote_reports_actual_discrepancy(workspace):
    verifier = support_verifier(workspace)
    ticket(workspace)
    verifier.provider = FakeProvider([SummaryAssessment(accurate=True, reason='Accurate summary', source_quotes=['No outage occurred'], contradictions=[])])
    result = await verify(verifier)
    assert not result.verified
    assert any('quotes' in discrepancy for discrepancy in result.discrepancies)
