"""Bounded local CSV profiling; full data never enters decision context."""
import hashlib
import json
import pandas as pd
from backend.app.capabilities.documents import resolve_file

MAX_ROWS = 200_000
MAX_COLUMNS = 64


def load_dataset(document_id):
    _, path = resolve_file(document_id)
    if path.suffix.lower() != '.csv':
        raise ValueError('Dataset profiling supports registered CSV files')
    frame = pd.read_csv(path, nrows=MAX_ROWS + 1)
    if len(frame) > MAX_ROWS or len(frame.columns) > MAX_COLUMNS or frame.empty:
        raise ValueError('CSV must contain 1–200000 rows and at most 64 columns')
    if any(len(str(column)) > 60 for column in frame.columns):
        raise ValueError('Column names exceed the supported length')
    return path, frame


def preview(frame, limit=3):
    sample = frame.head(limit).copy()
    for column in sample.select_dtypes(include=['object', 'string']).columns:
        sample[column] = sample[column].map(lambda value: str(value)[:160] if pd.notna(value) else None)
    return json.loads(sample.to_json(orient='records', date_format='iso'))


def dataset_profile(document_id):
    path, frame = load_dataset(document_id)
    numeric = frame.select_dtypes(include='number').describe().round(6)
    summary = json.loads(numeric.to_json()) if len(numeric.columns) else {}
    categorical = {column: [str(value)[:160] for value in frame[column].dropna().drop_duplicates().head(5)]
                   for column in frame.select_dtypes(exclude='number').columns}
    return {'document_id': document_id, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'shape': list(frame.shape), 'columns': list(frame.columns), 'dtypes': {key: str(value) for key, value in frame.dtypes.items()},
            'sample_rows': preview(frame), 'null_counts': {key: int(value) for key, value in frame.isna().sum().items()},
            'unique_counts': {key: int(value) for key, value in frame.nunique(dropna=True).items()},
            'numeric_summaries': summary, 'categorical_values': categorical}


def context_profile(profile, objective=''):
    columns = list(profile['dtypes'])
    columns.sort(key=lambda column: (column.casefold() not in objective.casefold(), profile['columns'].index(column)))
    selected = columns[:12]
    while True:
        result = {'document_id': profile['document_id'], 'sha256': profile['sha256'], 'shape': profile['shape'],
                  'column_types': profile['dtypes'], 'sample_columns': selected,
                  'sample_rows': [{key: str(value)[:60] if isinstance(value, str) else value for key, value in row.items() if key in selected} for row in profile['sample_rows']],
                  'null_counts': {key: value for key, value in profile['null_counts'].items() if key in selected},
                  'unique_counts': {key: value for key, value in profile['unique_counts'].items() if key in selected},
                  'numeric_summaries': {key: {stat: value for stat, value in summary.items() if stat in ('count', 'min', 'max', 'mean')} for key, summary in profile['numeric_summaries'].items() if key in selected},
                  'categorical_values': {key: [value[:40] for value in values[:3]] for key, values in profile['categorical_values'].items() if key in selected}}
        if len(json.dumps(result)) <= 8000 or not selected:
            return result
        selected = selected[:-1]
