"""
SQLite database management for Operon workspace.
Uses parameterized queries, WAL mode, foreign keys, and strict constraints.
"""

import os
import sqlite3
from pathlib import Path
from contextlib import contextmanager

DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "workspace.db"

def get_db_path() -> Path:
    override = os.getenv("OPERON_DB_PATH")
    if override:
        return Path(override)
    return DEFAULT_DB_PATH

def init_db(db_path: Path | None = None) -> None:
    path = db_path or get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with sqlite3.connect(str(path)) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Finance Invoices
        conn.execute("""
            CREATE TABLE IF NOT EXISTS finance_invoices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                invoice_number TEXT NOT NULL,
                amount_minor INTEGER NOT NULL,
                currency TEXT NOT NULL DEFAULT 'INR',
                due_date TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Pending',
                created_at TEXT NOT NULL,
                source_reference TEXT,
                CONSTRAINT uq_company_invoice UNIQUE (company, invoice_number)
            )
        """)
        
        # CRM Accounts
        conn.execute("""
            CREATE TABLE IF NOT EXISTS crm_accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_name TEXT NOT NULL UNIQUE,
                tier TEXT NOT NULL,
                mrr INTEGER NOT NULL DEFAULT 0,
                account_manager TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Active',
                created_at TEXT NOT NULL
            )
        """)
        
        # Support Tickets
        conn.execute("""
            CREATE TABLE IF NOT EXISTS support_tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT NOT NULL UNIQUE,
                customer TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Open',
                source_reference TEXT,
                summary TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Documents Index
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                filepath TEXT NOT NULL,
                title TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                company TEXT,
                doc_date TEXT,
                content_preview TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Runs table (Agent executions)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                objective TEXT NOT NULL,
                status TEXT NOT NULL,
                plan TEXT,
                working_memory TEXT,
                verification_result TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Run Events table (Timeline & SSE streaming)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS run_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs (run_id) ON DELETE CASCADE
            )
        """)
        
        conn.execute("CREATE TABLE IF NOT EXISTS workspace_lease(singleton INTEGER PRIMARY KEY CHECK(singleton=1),run_id TEXT NOT NULL,token TEXT NOT NULL)")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(support_tickets)")}
        if "mutation_key" not in columns:
            conn.execute("ALTER TABLE support_tickets ADD COLUMN mutation_key TEXT")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ticket_mutation_identity ON support_tickets(mutation_key) WHERE mutation_key IS NOT NULL")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS finance_canonical_identity ON finance_invoices(LOWER(TRIM(company)),LOWER(TRIM(invoice_number)))")
        conn.commit()

@contextmanager
def get_db_connection(db_path: Path | None = None, read_only: bool = False):
    """Yields an SQLite connection configured with Row factory and proper pragmas."""
    path = db_path or get_db_path()
    if read_only:
        uri = f"file:{path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(str(path))
        conn.execute("PRAGMA foreign_keys = ON")
    
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
