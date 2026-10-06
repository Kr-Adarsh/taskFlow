"""Registered-file inspection and provenance-preserving local chunk retrieval."""
import hashlib
import re
from pathlib import Path
from backend.app.tools.base import ToolResult
from backend.app.tools.document_tools import document_search
from backend.app.workspace.service import get_document_by_filename, extract_document_text

MAX_FILE_BYTES = 20 * 1024 * 1024


def resolve_file(document_id):
    doc = get_document_by_filename(document_id)
    if not doc:
        raise ValueError('Input must be an exact registered document ID')
    path = Path(doc.filepath).resolve(strict=True)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError('Input exceeds the supported file size')
    return doc, path


def chunks(document_id):
    doc, path = resolve_file(document_id)
    if path.suffix.lower() == '.csv':
        raise ValueError('Profile datasets instead of reading raw CSV into model context')
    if path.suffix.lower() == '.pdf':
        from pypdf import PdfReader
        sections = [(str(i + 1), page.extract_text() or '') for i, page in enumerate(PdfReader(path).pages)]
    else:
        sections = [('1', extract_document_text(path))]
    result = []
    for page, text in sections:
        for start in range(0, len(text), 1000):
            part = text[start:start + 1200]
            result.append({'document_id': doc.filename, 'page': page, 'section': f'characters {start}-{start+len(part)}',
                           'chunk_id': f'{doc.filename}:{page}:{start}', 'text': part})
    return result


def inspect_file(document_id):
    try:
        doc, path = resolve_file(document_id)
        data = {'document_id': doc.filename, 'file_type': path.suffix.lower(), 'bytes': path.stat().st_size,
                'title': doc.title, 'doc_type': doc.doc_type, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if path.suffix.lower() != '.csv':
            data['chunks'] = [{key: value for key, value in item.items() if key != 'text'} for item in chunks(document_id)][:100]
        return ToolResult(ok=True, data=data, evidence={'document_id': doc.filename})
    except (ValueError, OSError) as error:
        return ToolResult(ok=False, error=str(error), error_code='FILE_UNAVAILABLE')


def search_documents(query):
    result = document_search(query)
    if result.ok:
        for doc in result.data['documents']:
            doc['document_id'] = doc.pop('filename')
        result.data['documents'] = result.data['documents'][:12]
    return result


def read_document_chunks(document_id, query='', chunk_ids=None, limit=3):
    try:
        available = chunks(document_id)
        if chunk_ids:
            requested = set(chunk_ids)
            available = [chunk for chunk in available if chunk['chunk_id'] in requested]
            if len(available) != len(requested):
                raise ValueError('Unknown source chunk IDs')
        elif query:
            terms = set(re.findall(r'\w+', query.casefold()))
            available.sort(key=lambda item: len(terms & set(re.findall(r'\w+', item['text'].casefold()))), reverse=True)
        selected = available[:limit]
        return ToolResult(ok=True, data={'document_id': document_id, 'chunks': selected, 'returned_chunks': len(selected)},
                          evidence={'document_id': document_id, 'chunk_ids': [chunk['chunk_id'] for chunk in selected]})
    except (ValueError, OSError) as error:
        return ToolResult(ok=False, error=str(error), error_code='DOCUMENT_UNAVAILABLE')


def register_documents(registry):
    from backend.app.capabilities.registry import schema
    registry.add('documents', 'inspect_file', 'Inspect a registered file and discover its type, size and source chunk IDs.', schema(document_id={'type': 'string'}), inspect_file)
    registry.add('documents', 'search_documents', 'Search registered documents by terms; returns candidate identities and metadata, not complete documents.', schema(query={'type': 'string'}), search_documents)
    registry.add('documents', 'read_document_chunks', 'Retrieve bounded source chunks locally. Use query or exact chunk IDs; every chunk retains provenance. CSV data must use profile_dataset instead.',
                 {'type': 'object', 'properties': {'document_id': {'type': 'string'}, 'query': {'type': 'string'}, 'chunk_ids': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 5}, 'limit': {'type': 'integer', 'minimum': 1, 'maximum': 5}}, 'required': ['document_id']}, read_document_chunks)
