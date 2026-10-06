"""
Unit tests for generic document search and read tools.
"""

from pathlib import Path
import pytest

from backend.app.workspace.seed import seed_workspace
from backend.app.tools.document_tools import document_search, document_read

@pytest.fixture(autouse=True)
def setup_docs():
    seed_workspace()

def test_document_search_acme():
    res = document_search("Acme")
    assert res.ok is True
    data = res.data
    assert data["total_matches"] >= 2
    filenames = [d["filename"] for d in data["documents"]]
    assert "acme_invoice_1044.pdf" in filenames
    assert "acme_invoice_1021.pdf" in filenames

def test_document_search_complaint():
    res = document_search("4821")
    assert res.ok is True
    data = res.data
    assert data["total_matches"] >= 1
    assert data["documents"][0]["filename"] == "complaint_4821.txt"

def test_document_read_pdf():
    res = document_read("acme_invoice_1044.pdf")
    assert res.ok is True
    assert res.data["filename"] == "acme_invoice_1044.pdf"
    content = res.data["content"]
    assert "INV-1044" in content
    assert "84,500" in content
    assert "2026-09-15" in content

def test_document_read_txt():
    res = document_read("complaint_4821.txt")
    assert res.ok is True
    content = res.data["content"]
    assert "Acme Corp" in content
    assert "Critical Outage" in content

def test_document_read_not_found():
    res = document_read("non_existent_doc.pdf")
    assert res.ok is False
    assert res.error_code == "DOCUMENT_NOT_FOUND"
