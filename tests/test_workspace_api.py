"""
Focused API and HTML integration tests for workspace applications.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.workspace.seed import reset_demo_env

@pytest.fixture(autouse=True)
def reset_db_before_test():
    reset_demo_env()
    yield

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_workspace_html_pages_render():
    for page in ["/workspace/finance", "/workspace/crm", "/workspace/support", "/workspace/documents"]:
        res = client.get(page)
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        assert "TaskFlow Workspace" in res.text

def test_finance_form_submission_success():
    data = {
        "company": "Acme Corp",
        "invoice_number": "INV-1044",
        "amount": "84,500",
        "currency": "INR",
        "source_reference": "acme_invoice_1044.pdf",
        "due_date": "2026-10-15"
    }
    res = client.post("/workspace/finance/submit", data=data)
    assert res.status_code == 200
    assert "recorded successfully" in res.text
    assert "INV-1044" in res.text

def test_finance_form_duplicate_replacement():
    data = {
        "company": "Acme Corp",
        "invoice_number": "INV-1005",  # Already in initial seed
        "amount": "30,000",
        "currency": "INR",
        "source_reference": "acme_invoice_1005.pdf",
        "due_date": "2026-07-31"
    }
    res = client.post("/workspace/finance/submit", data=data)
    assert res.status_code == 200
    assert "recorded successfully" in res.text
    invoices = client.get('/api/workspace/invoices?company=Acme%20Corp').json()
    row, = invoices
    assert row['invoice_number'] == 'INV-1005'
    assert row['source_reference'] == data['source_reference']
    assert row['updated_at'] and row['status'] == 'Paid'

def test_finance_fault_injection_on_api():
    # Arm fault
    client.post("/api/workspace/fault-injection", json={"target": "finance_create_invoice", "count": 1})

    payload = {
        "company": "Acme Corp",
        "invoice_number": "INV-2001",
        "amount": "12,000",
        "currency": "INR",
        "due_date": "2026-11-15"
    }

    # First attempt: 503
    res1 = client.post("/api/workspace/invoices", json=payload)
    assert res1.status_code == 503
    data1 = res1.json()
    assert data1["retriable"] is True
    assert "Finance Service Unavailable" in data1["error"]

    # Second attempt: 200 OK
    res2 = client.post("/api/workspace/invoices", json=payload)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["ok"] is True
    assert data2["data"]["invoice_number"] == "INV-2001"

def test_crm_search_api():
    res = client.get("/api/workspace/crm?search=Acme")
    assert res.status_code == 200
    results = res.json()
    assert len(results) == 1
    assert results[0]["customer_name"] == "Acme Corp"
    assert results[0]["tier"] == "Enterprise"

def test_support_ticket_creation():
    payload = {
        "customer": "Acme Corp",
        "priority": "High",
        "summary": "Urgent outage on EU server cluster",
        "source_reference": "complaint_4821"
    }
    res = client.post("/api/workspace/tickets", json=payload)
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["customer"] == "Acme Corp"
    assert data["priority"] == "High"
    assert data["source_reference"] == "complaint_4821.txt"

def test_all_supported_currency_options_and_source_visible():
    from backend.app.workspace.models import format_minor_to_money
    assert format_minor_to_money(12300,'EUR')=='€123.00'
    assert format_minor_to_money(12300,'GBP')=='£123.00'
    page=client.get('/workspace/finance')
    for currency in ['INR','USD','EUR','GBP']:
        assert f'value="{currency}"' in page.text
    assert 'acme_invoice_1005.txt' in page.text
