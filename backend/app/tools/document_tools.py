"""
Generic document search and reading tools for TaskFlow.
Operates on the workspace document library fixtures.
"""

import re
from typing import Optional
from backend.app.tools.base import ToolResult
from backend.app.workspace.service import list_documents, get_document_by_filename, extract_document_text

def document_search(query: str) -> ToolResult:
    """
    Search the company document repository for relevant invoices, contracts, or complaints.
    Returns matched documents with metadata and content preview.
    """
    if not query or not query.strip():
        return ToolResult(ok=False, error="Search query cannot be empty", error_code="INVALID_QUERY")
    
    q = query.lower().strip()
    query_tokens = [tok for tok in re.findall(r"\w+", q) if len(tok) >= 2]
    all_docs = list_documents()
    scored_matches: list[tuple[int, str, dict]] = []
    
    for d in all_docs:
        target_text = f"{d.filename} {d.title} {d.company or ''} {d.doc_date or ''} {d.content_preview or ''}".lower()
        if q in target_text:
            score = 100
        elif query_tokens and all(tok in target_text for tok in query_tokens):
            score = 80
        elif query_tokens:
            matched_count = sum(1 for tok in query_tokens if tok in target_text)
            score = matched_count * 10
        else:
            score = 0

        if score > 0:
            scored_matches.append((score, d.doc_date or "", {
                "filename": d.filename,
                "title": d.title,
                "company": d.company,
                "doc_date": d.doc_date,
                "doc_type": d.doc_type,
                "preview": d.content_preview[:150] if d.content_preview else ""
            }))
            
    # Sort matches by score descending, then doc_date descending
    scored_matches.sort(key=lambda x: (x[0], x[1]), reverse=True)
    matches = [item[2] for item in scored_matches]
    
    return ToolResult(
        ok=True,
        data={
            "query": query,
            "total_matches": len(matches),
            "documents": matches
        },
        evidence={"searched_query": query, "match_count": len(matches)}
    )

def document_read(filename: str) -> ToolResult:
    """
    Reads the complete text of a document from the company document library.
    Supports both text files and PDF invoices/documents.
    """
    if not filename or not filename.strip():
        return ToolResult(ok=False, error="Filename must be provided", error_code="MISSING_FILENAME")
    
    doc = get_document_by_filename(filename.strip())
    if not doc:
        return ToolResult(
            ok=False,
            error=f"Document '{filename}' not found in company repository",
            error_code="DOCUMENT_NOT_FOUND"
        )
    
    try:
        content = extract_document_text(doc.filepath)
        return ToolResult(
            ok=True,
            data={
                "filename": doc.filename,
                "title": doc.title,
                "company": doc.company,
                "doc_date": doc.doc_date,
                "doc_type": doc.doc_type,
                "content": content
            },
            evidence={
                "filename": doc.filename,
                "source_path": doc.filepath,
                "doc_date": doc.doc_date
            }
        )
    except Exception as e:
        return ToolResult(
            ok=False,
            error=f"Failed to extract document text from {filename}: {str(e)}",
            error_code="EXTRACTION_FAILED"
        )
