"""
Workspace API endpoints and interactive HTML pages for Finance, CRM, Support, and Documents.
Designed for both human use and autonomous Playwright browser interaction.
"""

from pathlib import Path
from html import escape
from types import SimpleNamespace
from urllib.parse import quote
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from backend.app.workspace.models import (
    InvoiceCreate,
    SupportTicketCreate,
    format_minor_to_money,
)
from backend.app.workspace.fault_injection import fault_manager, FaultInjectionError
from backend.app.workspace.seed import seed_workspace, reset_demo_env
from backend.app.workspace.service import (
    create_invoice,
    list_invoices,
    get_invoice,
    list_crm_accounts,
    get_crm_account_by_name,
    search_crm_accounts,
    create_support_ticket,
    list_support_tickets,
    get_support_ticket_by_id,
    list_documents,
    get_document_by_filename,
    extract_document_text,
)

router = APIRouter()


def html_record(record):
    return SimpleNamespace(**{key: escape(value, quote=True) if isinstance(value, str) else value for key, value in record.model_dump().items()})


# --- REST Endpoints ---

@router.get("/api/workspace/invoices")
def api_list_invoices(company: Optional[str] = None):
    return [inv.model_dump() for inv in list_invoices(company=company)]

@router.post("/api/workspace/invoices")
def api_create_invoice(payload: InvoiceCreate):
    try:
        inv = create_invoice(payload)
        return {"ok": True, "data": inv.model_dump()}
    except FaultInjectionError as e:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "error": e.message,
                "error_code": e.error_code,
                "retriable": e.retriable
            }
        )
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": str(e), "error_code": "VALIDATION_OR_DUPLICATE", "retriable": False}
        )

@router.get("/api/workspace/crm")
def api_list_crm(search: Optional[str] = None):
    if search:
        accounts = search_crm_accounts(search)
    else:
        accounts = list_crm_accounts()
    return [acc.model_dump() for acc in accounts]

@router.get("/api/workspace/crm/{customer_name}")
def api_get_crm(customer_name: str):
    acc = get_crm_account_by_name(customer_name)
    if not acc:
        raise HTTPException(status_code=404, detail="Customer not found")
    return acc.model_dump()

@router.get("/api/workspace/tickets")
def api_list_tickets(customer: Optional[str] = None):
    return [t.model_dump() for t in list_support_tickets(customer=customer)]

@router.post("/api/workspace/tickets")
def api_create_ticket(payload: SupportTicketCreate):
    try:
        ticket = create_support_ticket(payload)
        return {"ok": True, "data": ticket.model_dump()}
    except ValueError as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e), "retriable": False})

@router.get("/api/workspace/documents")
def api_list_documents():
    return [d.model_dump() for d in list_documents()]

@router.get("/api/workspace/documents/{filename}/content")
def api_get_document_content(filename: str):
    doc = get_document_by_filename(filename)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if Path(doc.filepath).suffix.lower() == '.csv':
        from backend.app.capabilities.python.profile import dataset_profile
        return {"filename": doc.filename, "profile": dataset_profile(doc.filename)}
    text = extract_document_text(doc.filepath)
    return {"filename": doc.filename, "content": text}

@router.post("/api/workspace/reset")
def api_reset():
    reset_demo_env()
    return {"ok": True, "message": "Demo environment reset to baseline seed."}

class FaultInjectionPayload(BaseModel):
    target: str = "finance_create_invoice"
    count: int = 1

@router.post("/api/workspace/fault-injection")
def api_arm_fault(payload: FaultInjectionPayload):
    fault_manager.arm(payload.target, payload.count)
    return {"ok": True, "message": f"Fault injected for {payload.target} ({payload.count} times)."}

# --- Workspace Semantic HTML Pages for Playwright & Browser Inspection ---

WORKSPACE_LAYOUT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - TaskFlow Workspace</title>
    <script src="/static/theme.js"></script>
    <link rel="stylesheet" href="/static/theme.css">
</head>
<body class="workspace">
    <div class="container">
        <div class="nav-tabs">
            <a href="/workspace/finance" id="tab_finance" class="{active_finance}">Finance</a>
            <a href="/workspace/crm" id="tab_crm" class="{active_crm}">CRM</a>
            <a href="/workspace/support" id="tab_support" class="{active_support}">Support</a>
            <a href="/workspace/documents" id="tab_documents" class="{active_documents}">Documents</a>
        </div>
        {content}
    </div>
</body>
</html>
"""

@router.get("/workspace/finance", response_class=HTMLResponse)
def page_finance(
    message: Optional[str] = None,
    error: Optional[str] = None,
    status_code: Optional[int] = None,
    values: Optional[dict] = None
):
    invoices = list_invoices()
    v = {key: escape(str(value), quote=True) for key, value in (values or {}).items()}
    error = escape(error, quote=True) if error else None
    message = escape(message, quote=True) if message else None
    
    alert_html = ""
    if error:
        alert_html = f'<div id="feedback_message" class="alert alert-error" data-status="{status_code or 400}">Error: {error}</div>'
    elif message:
        alert_html = f'<div id="feedback_message" class="alert alert-success" data-status="200">{message}</div>'
        
    rows_html = ""
    for inv in invoices:
        inv = html_record(inv)
        rows_html += f"""
        <tr id="invoice_row_{inv.id}">
            <td id="invoice_company_{inv.id}">{inv.company}</td>
            <td id="invoice_number_{inv.id}"><strong>{inv.invoice_number}</strong></td>
            <td id="invoice_amount_{inv.id}">{inv.amount_formatted} {inv.currency}</td>
            <td>{inv.source_reference or "-"}</td>
            <td id="invoice_due_date_{inv.id}">{inv.due_date}</td>
            <td id="invoice_status_{inv.id}">{inv.status}</td>
            <td id="invoice_created_{inv.id}"><small>{inv.created_at[:19]}</small></td>
        </tr>
        """
        
    currency_options = "".join(f'<option value="{currency}" {"selected" if v.get("currency", "INR") == currency else ""}>{currency}</option>' for currency in ("INR", "USD", "EUR", "GBP"))
    content = f"""
    <h1>Finance - Invoices Ledger</h1>
    {alert_html}
    <div class="card">
        <h3>Record New Invoice</h3>
        <form method="POST" action="/workspace/finance/submit" id="form_create_invoice">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                <div class="form-group">
                    <label for="company">Company / Vendor</label>
                    <input type="text" id="company" name="company" placeholder="Acme Corp" value="{v.get('company', '')}" required />
                </div>
                <div class="form-group">
                    <label for="invoice_number">Invoice Number</label>
                    <input type="text" id="invoice_number" name="invoice_number" placeholder="INV-1044" value="{v.get('invoice_number', '')}" required />
                </div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px;">
                <div class="form-group">
                    <label for="amount">Total Amount</label>
                    <input type="text" id="amount" name="amount" placeholder="84,500" value="{v.get('amount', '')}" required />
                </div>
                <div class="form-group">
                    <label for="currency">Currency</label>
                    <select id="currency" name="currency">
                        {currency_options}
                    </select>
                </div>
                <div class="form-group">
                    <label for="due_date">Due Date (YYYY-MM-DD)</label>
                    <input type="date" id="due_date" name="due_date" value="{v.get('due_date', '')}" required />
                </div>
            </div>
            <div class="form-group"><label for="source_reference">Source Document Filename</label><input id="source_reference" name="source_reference" value="{v.get('source_reference', '')}" required /></div>
            <button type="submit" id="submit_invoice">Record Invoice</button>
        </form>
    </div>
    
    <div class="card">
        <h3>Invoices ({len(invoices)})</h3>
        <table>
            <thead>
                <tr>
                    <th>Company</th>
                    <th>Invoice #</th>
                    <th>Amount / Currency</th>
                    <th>Source</th>
                    <th>Due Date</th>
                    <th>Status</th>
                    <th>Recorded</th>
                </tr>
            </thead>
            <tbody id="invoices_table_body">
                {rows_html}
            </tbody>
        </table>
    </div>
    """
    
    return WORKSPACE_LAYOUT_TEMPLATE.format(
        title="Finance",
        active_finance="active",
        active_crm="",
        active_support="",
        active_documents="",
        content=content
    )

@router.post("/workspace/finance/submit", response_class=HTMLResponse)
def form_submit_finance(
    company: str = Form(...),
    invoice_number: str = Form(...),
    amount: str = Form(...),
    currency: str = Form("INR"),
    due_date: str = Form(...),
    source_reference: str = Form(...)
):
    form_vals = {
        "company": company,
        "invoice_number": invoice_number,
        "amount": amount,
        "currency": currency,
        "due_date": due_date,
        "source_reference": source_reference
    }
    try:
        inv = create_invoice(
            InvoiceCreate(
                company=company,
                invoice_number=invoice_number,
                amount=amount,
                currency=currency,
                due_date=due_date,
                source_reference=source_reference
            )
        )
        msg = f"Invoice {inv.invoice_number} for {inv.company} ({inv.amount_formatted}) recorded successfully."
        return page_finance(message=msg)
    except FaultInjectionError as e:
        return HTMLResponse(page_finance(error=e.message, status_code=503, values=form_vals), status_code=503)
    except ValueError as e:
        return HTMLResponse(page_finance(error=str(e), status_code=400, values=form_vals), status_code=400)

@router.get("/workspace/crm", response_class=HTMLResponse)
def page_crm(search: Optional[str] = None):
    accounts = search_crm_accounts(search) if search else list_crm_accounts()
    
    rows_html = ""
    search = escape(search, quote=True) if search else None
    for acc in accounts:
        acc = html_record(acc)
        badge_class = f"badge-{acc.tier.lower()}"
        rows_html += f"""
        <tr id="crm_row_{acc.id}">
            <td id="crm_name_{acc.id}"><strong>{acc.customer_name}</strong></td>
            <td id="crm_tier_{acc.id}"><span class="badge {badge_class}">{acc.tier}</span></td>
            <td id="crm_manager_{acc.id}">{acc.account_manager}</td>
            <td id="crm_status_{acc.id}">{acc.status}</td>
        </tr>
        """
        
    content = f"""
    <h1>CRM - Customer Accounts</h1>
    <div class="card">
        <form method="GET" action="/workspace/crm" id="form_crm_search">
            <div style="display: flex; gap: 12px;">
                <input type="text" id="search_crm" name="search" placeholder="Search customer (e.g. Acme Corp)..." value="{search or ''}" />
                <button type="submit" id="button_search_crm">Search</button>
                {f'<a href="/workspace/crm" style="color: var(--text-secondary); align-self: center; margin-left: 8px;">Clear</a>' if search else ''}
            </div>
        </form>
    </div>
    
    <div class="card">
        <h3>Accounts Directory ({len(accounts)})</h3>
        <table>
            <thead>
                <tr>
                    <th>Customer Name</th>
                    <th>Account Tier</th>
                    <th>Account Manager</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody id="crm_table_body">
                {rows_html}
            </tbody>
        </table>
    </div>
    """
    
    return WORKSPACE_LAYOUT_TEMPLATE.format(
        title="CRM",
        active_finance="",
        active_crm="active",
        active_support="",
        active_documents="",
        content=content
    )

@router.get("/workspace/support", response_class=HTMLResponse)
def page_support(message: Optional[str] = None, error: Optional[str] = None):
    tickets = list_support_tickets()
    error = escape(error, quote=True) if error else None
    message = escape(message, quote=True) if message else None
    
    alert_html = ""
    if error:
        alert_html = f'<div id="feedback_message" class="alert alert-error">{error}</div>'
    elif message:
        alert_html = f'<div id="feedback_message" class="alert alert-success">{message}</div>'
        
    rows_html = ""
    for t in tickets:
        t = html_record(t)
        badge_class = f"badge-{t.priority.lower()}"
        rows_html += f"""
        <tr id="ticket_row_{t.id}">
            <td id="ticket_id_{t.id}"><strong>{t.ticket_id}</strong></td>
            <td id="ticket_customer_{t.id}">{t.customer}</td>
            <td id="ticket_priority_{t.id}"><span class="badge {badge_class}">{t.priority}</span></td>
            <td>{t.source_reference or "-"}</td>
            <td id="ticket_summary_{t.id}">{t.summary}</td>
            <td id="ticket_status_{t.id}">{t.status}</td>
        </tr>
        """
        
    content = f"""
    <h1>Support - Customer Tickets</h1>
    {alert_html}
    <div class="card">
        <h3>Create Support Ticket</h3>
        <form method="POST" action="/workspace/support/submit" id="form_create_ticket">
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px;">
                <div class="form-group">
                    <label for="ticket_customer">Customer Name</label>
                    <input type="text" id="ticket_customer" name="customer" placeholder="Acme Corp" required />
                </div>
                <div class="form-group">
                    <label for="ticket_priority">Priority</label>
                    <select id="ticket_priority" name="priority">
                        <option value="High">High</option>
                        <option value="Medium" selected>Medium</option>
                        <option value="Low">Low</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="ticket_source_ref">Source Document Filename</label>
                    <input type="text" id="ticket_source_ref" name="source_reference" placeholder="Source document filename" required />
                </div>
            </div>
            <div class="form-group">
                <label for="ticket_summary">Summary / Description</label>
                <textarea id="ticket_summary" name="summary" rows="3" placeholder="Summarize the customer complaint..." required></textarea>
            </div>
            <button type="submit" id="submit_ticket">Create Ticket</button>
        </form>
    </div>
    
    <div class="card">
        <h3>Tickets ({len(tickets)})</h3>
        <table>
            <thead>
                <tr>
                    <th>Ticket #</th>
                    <th>Customer</th>
                    <th>Priority</th>
                    <th>Source</th>
                    <th>Summary</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody id="tickets_table_body">
                {rows_html}
            </tbody>
        </table>
    </div>
    """
    
    return WORKSPACE_LAYOUT_TEMPLATE.format(
        title="Support",
        active_finance="",
        active_crm="",
        active_support="active",
        active_documents="",
        content=content
    )

@router.post("/workspace/support/submit", response_class=HTMLResponse)
def form_submit_support(
    customer: str = Form(...),
    priority: str = Form("Medium"),
    summary: str = Form(...),
    source_reference: str = Form(...)
):
    try:
        ticket = create_support_ticket(
            SupportTicketCreate(
                customer=customer,
                priority=priority,
                summary=summary,
                source_reference=source_reference
            )
        )
        msg = f"Support ticket {ticket.ticket_id} for {ticket.customer} created successfully."
        return page_support(message=msg)
    except ValueError as e:
        return HTMLResponse(page_support(error=str(e)), status_code=400)

@router.get("/workspace/documents", response_class=HTMLResponse)
def page_documents(view: Optional[str] = None):
    docs = list_documents()
    
    preview_box = ""
    if view:
        doc = get_document_by_filename(view)
        if doc:
            text = 'CSV dataset. Use profile_dataset to inspect its schema and execute_python for local analysis; full data stays local.' if Path(doc.filepath).suffix.lower() == '.csv' else extract_document_text(doc.filepath)
            content_text = escape(text[:12000], quote=True)
            doc = html_record(doc)
            preview_box = f"""
            <div class="card" id="doc_preview_card" style="border: 1px solid var(--accent);">
                <h3>Viewing Document: {doc.filename}</h3>
                <p><strong>Title:</strong> {doc.title} | <strong>Date:</strong> {doc.doc_date} | <strong>Company:</strong> {doc.company}</p>
                <pre id="doc_full_text" style="background: #0b132b; padding: 16px; border-radius: 6px; white-space: pre-wrap; font-family: monospace; color: #e2e8f0;">{content_text}</pre>
                <a href="/workspace/documents" style="color: var(--accent);">Close Preview</a>
            </div>
            """
            
    rows_html = ""
    for d in docs:
        view_url = escape(quote(d.filename, safe=""), quote=True)
        d = html_record(d)
        rows_html += f"""
        <tr id="doc_row_{d.id}">
            <td id="doc_title_{d.id}"><strong>{d.title}</strong></td>
            <td id="doc_filename_{d.id}"><code>{d.filename}</code></td>
            <td id="doc_company_{d.id}">{d.company or '-'}</td>
            <td id="doc_date_{d.id}">{d.doc_date or '-'}</td>
            <td><a href="/workspace/documents?view={view_url}" id="btn_view_{d.filename}">View Content</a></td>
        </tr>
        """
        
    content = f"""
    <h1>Documents Library</h1>
    {preview_box}
    <div class="card">
        <h3>Available Documents ({len(docs)})</h3>
        <table>
            <thead>
                <tr>
                    <th>Title</th>
                    <th>Filename</th>
                    <th>Associated Company</th>
                    <th>Date</th>
                    <th>Action</th>
                </tr>
            </thead>
            <tbody id="documents_table_body">
                {rows_html}
            </tbody>
        </table>
    </div>
    """
    
    return WORKSPACE_LAYOUT_TEMPLATE.format(
        title="Documents",
        active_finance="",
        active_crm="",
        active_support="",
        active_documents="active",
        content=content
    )
