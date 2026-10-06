"""
Application service functions for simulated company workspace.
Provides domain logic, transactional mutations, constraints, and fault injection hooks.
"""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import re
from backend.app.workspace.lease import assert_mutation_access
import pypdf

from backend.app.workspace.db import get_db_connection, get_db_path
from backend.app.workspace.fault_injection import fault_manager
from backend.app.workspace.models import (
    InvoiceCreate,
    InvoiceRecord,
    CRMAccountRecord,
    SupportTicketCreate,
    SupportTicketRecord,
    DocumentItem,
    parse_money_to_minor,
    format_minor_to_money,
)

# --- Finance Invoices ---

def create_invoice(invoice_data: InvoiceCreate, db_path: Path | None = None) -> InvoiceRecord:
    """
    Records an invoice, replacing submitted fields for an existing identity.
    Enforces deterministic fault injection before transaction commit.
    Enforces uniqueness of (company, invoice_number).
    """
    
    amount_minor = parse_money_to_minor(invoice_data.amount)
    with get_db_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        assert_mutation_access(conn)
        fault_manager.maybe_fail("finance_create_invoice")
        now_iso = datetime.now(timezone.utc).isoformat()
        existing = conn.execute(
            "SELECT * FROM finance_invoices WHERE LOWER(TRIM(company)) = LOWER(?) "
            "AND LOWER(TRIM(invoice_number)) = LOWER(?)",
            (invoice_data.company.strip(), invoice_data.invoice_number.strip()),
        ).fetchone()
        if existing:
            # Retain identity, creation time and payment status unless explicitly submitted.
            status = invoice_data.status if "status" in invoice_data.model_fields_set else existing["status"]
            conn.execute(
                "UPDATE finance_invoices SET amount_minor=?, currency=?, due_date=?, "
                "status=?, source_reference=?, updated_at=? WHERE id=?",
                (amount_minor, invoice_data.currency.upper(), invoice_data.due_date,
                 status, invoice_data.source_reference, now_iso, existing["id"]),
            )
            invoice_id = existing["id"]
        else:
            cursor = conn.execute(
                """
                INSERT INTO finance_invoices (
                    company, invoice_number, amount_minor, currency,
                    due_date, status, created_at, source_reference
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    invoice_data.company.strip(),
                    invoice_data.invoice_number.strip(),
                    amount_minor,
                    invoice_data.currency.upper(),
                    invoice_data.due_date,
                    invoice_data.status,
                    now_iso,
                    invoice_data.source_reference
                )
            )
            invoice_id = cursor.lastrowid
        conn.commit()

    return get_invoice(invoice_id, db_path=db_path)

def get_invoice(invoice_id: int, db_path: Path | None = None) -> InvoiceRecord | None:
    with get_db_connection(db_path, read_only=True) as conn:
        row = conn.execute("SELECT * FROM finance_invoices WHERE id = ?", (invoice_id,)).fetchone()
        if not row:
            return None
        return InvoiceRecord(
            id=row["id"],
            company=row["company"],
            invoice_number=row["invoice_number"],
            amount_minor=row["amount_minor"],
            amount_formatted=format_minor_to_money(row["amount_minor"], row["currency"]),
            currency=row["currency"],
            due_date=row["due_date"],
            status=row["status"],
            created_at=row["created_at"],
            source_reference=row["source_reference"],
            updated_at=row["updated_at"],
        )

def list_invoices(company: str | None = None, db_path: Path | None = None) -> list[InvoiceRecord]:
    with get_db_connection(db_path, read_only=True) as conn:
        if company:
            rows = conn.execute(
                "SELECT * FROM finance_invoices WHERE LOWER(company) = LOWER(?) ORDER BY id DESC",
                (company.strip(),)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM finance_invoices ORDER BY id DESC").fetchall()
            
        return [
            InvoiceRecord(
                id=r["id"],
                company=r["company"],
                invoice_number=r["invoice_number"],
                amount_minor=r["amount_minor"],
                amount_formatted=format_minor_to_money(r["amount_minor"], r["currency"]),
                currency=r["currency"],
                due_date=r["due_date"],
                status=r["status"],
                created_at=r["created_at"],
                source_reference=r["source_reference"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

# --- CRM Accounts ---

def list_crm_accounts(db_path: Path | None = None) -> list[CRMAccountRecord]:
    with get_db_connection(db_path, read_only=True) as conn:
        rows = conn.execute("SELECT * FROM crm_accounts ORDER BY customer_name ASC").fetchall()
        return [CRMAccountRecord(**dict(r)) for r in rows]

def get_crm_account_by_name(customer_name: str, db_path: Path | None = None) -> CRMAccountRecord | None:
    with get_db_connection(db_path, read_only=True) as conn:
        row = conn.execute(
            "SELECT * FROM crm_accounts WHERE LOWER(customer_name) = LOWER(?)",
            (customer_name.strip(),)
        ).fetchone()
        if not row:
            return None
        return CRMAccountRecord(**dict(row))

def search_crm_accounts(query: str, db_path: Path | None = None) -> list[CRMAccountRecord]:
    with get_db_connection(db_path, read_only=True) as conn:
        pattern = f"%{query.strip()}%"
        rows = conn.execute(
            "SELECT * FROM crm_accounts WHERE customer_name LIKE ? OR account_manager LIKE ? OR tier LIKE ?",
            (pattern, pattern, pattern)
        ).fetchall()
        return [CRMAccountRecord(**dict(r)) for r in rows]

# --- Support Tickets ---

def create_support_ticket(ticket_data: SupportTicketCreate, db_path: Path | None = None) -> SupportTicketRecord:
    now_iso = datetime.now(timezone.utc).isoformat()
    
    with get_db_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        assert_mutation_access(conn)
        reference = (ticket_data.source_reference or "").strip()
        if not reference:
            raise ValueError("Support ticket requires a stable source reference")
        matches = []
        for document in conn.execute("SELECT filename FROM documents_index"):
            filename = document["filename"]
            if reference.casefold() in (filename.casefold(), Path(filename).stem.casefold()) or (reference.isdigit() and re.search(rf"(?<!\d){re.escape(reference)}(?!\d)", filename)):
                matches.append(filename)
        if len(matches) > 1:
            raise ValueError("Source reference is ambiguous")
        if not matches:
            raise ValueError("Source reference must identify an existing document; use its exact filename")
        reference = matches[0]
        mutation_key = hashlib.sha256((ticket_data.customer.casefold() + "\0" + reference.casefold()).encode()).hexdigest()
        existing = conn.execute("SELECT * FROM support_tickets WHERE mutation_key=? OR (LOWER(customer)=LOWER(?) AND source_reference=?)", (mutation_key, ticket_data.customer, reference)).fetchall()
        if len(existing) > 1:
            raise ValueError("Duplicate source-linked tickets require manual correction")
        if existing:
            row = existing[0]
            if (row["priority"], row["summary"], row["status"]) != (ticket_data.priority, ticket_data.summary, ticket_data.status):
                raise ValueError("Source already has a different ticket; correction UI is unavailable")
            return SupportTicketRecord(**{key: row[key] for key in SupportTicketRecord.model_fields})
        seq = conn.execute("SELECT COALESCE(MAX(id),0)+1 FROM support_tickets").fetchone()[0]
        ticket_id = f"TIK-{8400 + seq}"
        cursor = conn.execute(
            """
            INSERT INTO support_tickets (
                ticket_id, customer, priority, status, source_reference, summary, created_at, mutation_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticket_id,
                ticket_data.customer.strip(),
                ticket_data.priority,
                ticket_data.status,
                reference,
                ticket_data.summary.strip(),
                now_iso,
                mutation_key
            )
        )
        rec_id = cursor.lastrowid
        conn.commit()

    return get_support_ticket_by_id(ticket_id, db_path=db_path)

def get_support_ticket_by_id(ticket_id: str, db_path: Path | None = None) -> SupportTicketRecord | None:
    with get_db_connection(db_path, read_only=True) as conn:
        row = conn.execute("SELECT * FROM support_tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
        if not row:
            return None
        return SupportTicketRecord(**{key: row[key] for key in SupportTicketRecord.model_fields})

def list_support_tickets(customer: str | None = None, db_path: Path | None = None) -> list[SupportTicketRecord]:
    with get_db_connection(db_path, read_only=True) as conn:
        if customer:
            rows = conn.execute(
                "SELECT * FROM support_tickets WHERE LOWER(customer) = LOWER(?) ORDER BY id DESC",
                (customer.strip(),)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM support_tickets ORDER BY id DESC").fetchall()
        return [SupportTicketRecord(**{key: r[key] for key in SupportTicketRecord.model_fields}) for r in rows]

# --- Document Extraction & Reading ---

def extract_document_text(filepath: Path | str) -> str:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Document file does not exist: {filepath}")
    
    if path.suffix.lower() == ".pdf":
        reader = pypdf.PdfReader(str(path))
        text_parts = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(text_parts).strip()
    else:
        return path.read_text(encoding="utf-8").strip()

def list_documents(db_path: Path | None = None) -> list[DocumentItem]:
    with get_db_connection(db_path, read_only=True) as conn:
        rows = conn.execute("SELECT * FROM documents_index ORDER BY doc_date DESC, id DESC").fetchall()
        return [DocumentItem(**dict(r)) for r in rows]

def get_document_by_filename(filename: str, db_path: Path | None = None) -> DocumentItem | None:
    with get_db_connection(db_path, read_only=True) as conn:
        row = conn.execute("SELECT * FROM documents_index WHERE LOWER(filename) = LOWER(?)", (filename.strip(),)).fetchone()
        if not row:
            return None
        return DocumentItem(**dict(row))
