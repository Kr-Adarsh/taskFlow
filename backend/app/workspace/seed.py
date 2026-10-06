"""
Deterministic seed and reset for Operon workspace.
Populates standard CRM accounts, baseline invoices, baseline support tickets,
and indexes fixture documents.
"""

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from backend.app.workspace.db import init_db, get_db_connection, get_db_path
from backend.app.workspace.lease import assert_mutation_access
from backend.app.workspace.fault_injection import fault_manager
from backend.app.workspace.service import extract_document_text

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "documents"

def seed_workspace(db_path: Path | None = None, force_reseed: bool = False) -> None:
    path = db_path or get_db_path()
    init_db(path)
    
    now_iso = datetime.now(timezone.utc).isoformat()
    
    with get_db_connection(path) as conn:
        if force_reseed:
            conn.execute("BEGIN IMMEDIATE")
            assert_mutation_access(conn, resetting=True)
            conn.execute("DELETE FROM finance_invoices")
            conn.execute("DELETE FROM crm_accounts")
            conn.execute("DELETE FROM support_tickets")
            conn.execute("DELETE FROM documents_index")
            conn.execute("DELETE FROM sqlite_sequence")
            conn.commit()

        # 1. Seed CRM Accounts
        crm_data = [
            ("Acme Corp", "Enterprise", 250000, "Sarah Jenkins", "Active", "2026-01-15T00:00:00Z"),
            ("Globex Inc", "Growth", 85000, "Bob Vance", "Active", "2026-02-10T00:00:00Z"),
            ("Beta Retail", "Starter", 15000, "Dave Miller", "Active", "2026-03-01T00:00:00Z"),
        ]
        for name, tier, mrr, manager, status, created in crm_data:
            conn.execute(
                """
                INSERT OR IGNORE INTO crm_accounts (customer_name, tier, mrr, account_manager, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (name, tier, mrr, manager, status, created)
            )

        # 2. Seed Baseline Finance Invoices
        finance_data = [
            ("Globex Inc", "GLX-101", 5000000, "INR", "2026-06-01", "Paid", "2026-06-01T10:00:00Z", "initial_seed"),
            ("Acme Corp", "INV-1005", 3000000, "INR", "2026-07-31", "Paid", "2026-07-01T09:00:00Z", "acme_invoice_1005.txt"),
        ]
        for comp, num, amt, cur, due, stat, created, ref in finance_data:
            conn.execute(
                """
                INSERT OR IGNORE INTO finance_invoices (company, invoice_number, amount_minor, currency, due_date, status, created_at, source_reference)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (comp, num, amt, cur, due, stat, created, ref)
            )

        # 3. Seed Baseline Support Tickets
        support_data = [
            ("TIK-1001", "Globex Inc", "Medium", "Resolved", "legacy_support", "Initial onboarding sync configuration", "2026-06-15T11:00:00Z"),
        ]
        for tid, cust, prio, stat, ref, summ, created in support_data:
            conn.execute(
                """
                INSERT OR IGNORE INTO support_tickets (ticket_id, customer, priority, status, source_reference, summary, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (tid, cust, prio, stat, ref, summ, created)
            )

        # 4. Index Fixture Documents
        doc_metadata = [
            ("acme_invoice_1021.pdf", "Acme Corporation Invoice INV-1021", "invoice", "Acme Corp", "2026-08-10"),
            ("acme_invoice_1044.pdf", "Acme Corporation Commercial Invoice INV-1044", "invoice", "Acme Corp", "2026-09-15"),
            ("acme_invoice_1005.txt", "Acme Corporation Invoice INV-1005", "invoice", "Acme Corp", "2026-07-01"),
            ("globex_invoice_301.pdf", "Globex Industries Invoice GLX-301", "invoice", "Globex Inc", "2026-09-20"),
            ("complaint_4821.txt", "Customer Complaint #4821 - Acme Corp", "complaint", "Acme Corp", "2026-10-02"),
            ("complaint_4822.txt", "Customer Complaint #4822 - Beta Retail", "complaint", "Beta Retail", "2026-10-03"),
            ("sales.csv", "Revenue by region and month (synthetic)", "dataset", None, "2026-09-30"),
        ]
        for fname, title, dtype, comp, ddate in doc_metadata:
            fpath = FIXTURES_DIR / fname
            if dtype == 'dataset':
                fpath = FIXTURES_DIR.parent / 'datasets' / fname
            preview = ""
            if fpath.exists() and dtype != 'dataset':
                try:
                    full_text = extract_document_text(fpath)
                    preview = full_text[:200]
                except Exception:
                    preview = ""
            conn.execute(
                """
                INSERT OR REPLACE INTO documents_index (filename, filepath, title, doc_type, company, doc_date, content_preview, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (fname, str(fpath), title, dtype, comp, ddate, preview, now_iso)
            )
            
        conn.commit()

def reset_demo_env(db_path: Path | None = None) -> None:
    """Resets the entire workspace environment to deterministic clean seed state."""
    seed_workspace(db_path=db_path, force_reseed=True)
    fault_manager.reset()
