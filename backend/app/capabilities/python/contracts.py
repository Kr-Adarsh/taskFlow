"""Shared generated-program contract; normalization never changes analysis code."""
import datetime
import math
from zoneinfo import ZoneInfo

IMPORTS = {'pandas', 'numpy', 'math', 'statistics', 'datetime', 'json'}
MAX_ATTEMPTS = 3
SUMMARY_LIMIT = 1000
METRIC_LIMIT = 32
METRIC_NAME_LIMIT = 80
METRIC_STRING_LIMIT = 200
TABLE_LIMIT = 8
TABLE_NAME_LIMIT = 60
UNKNOWN = object()

RESULT_SCHEMA = {
    'type': 'object', 'required': ['metrics'], 'additionalProperties': False,
    'properties': {
        'summary': {'type': 'string', 'maxLength': SUMMARY_LIMIT},
        'metrics': {'type': 'object', 'maxProperties': METRIC_LIMIT,
                    'propertyNames': {'maxLength': METRIC_NAME_LIMIT},
                    'additionalProperties': {'type': ['string', 'number', 'boolean', 'null'],
                                             'maxLength': METRIC_STRING_LIMIT}},
        'tables': {'type': 'object', 'maxProperties': TABLE_LIMIT,
                   'additionalProperties': {'python_type': 'pandas.DataFrame'}},
    },
}
RESULT_CONTRACT = {'canonical': RESULT_SCHEMA,
                   'shorthand': 'A dictionary of at most 32 named scalar metrics; no summary/metrics/tables keys.',
                   'scalars': 'Finite builtin/NumPy numbers, bool, str, null; date/datetime/Timestamp become ISO strings. No arrays, Series, nested metrics or custom objects.'}


class ResultError(ValueError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__(issues[0]['message'])


def issue(path, value, expected, message, **details):
    cls = type(value)
    return {'type': 'invalid_result_contract', 'path': path,
            'actual_type': type.__getattribute__(cls, '__module__') + '.' + type.__getattribute__(cls, '__name__'),
            'expected': expected, 'message': message, **details}


def scalar(value, path, np=None, pd=None, static=False, string_limit=METRIC_STRING_LIMIT):
    original = value
    if static and value is UNKNOWN:
        return value
    if pd is not None and (value is pd.NA or value is pd.NaT):
        return None
    if np is not None and type(value) in set(np.sctypeDict.values()):
        # Call the trusted NumPy implementation, never an arbitrary object's item method.
        value = np.generic.item(value)
        if type(value) is np.longdouble:
            converted = float(value)
            if not math.isfinite(converted) or value != np.longdouble(converted):
                raise ResultError([issue(path, original, 'exact finite JSON numeric scalar', 'Numeric conversion would lose precision or finiteness')])
            value = converted
    if type(value) in (datetime.datetime, datetime.date) or pd is not None and type(value) is pd.Timestamp:
        if type(value) is not datetime.date and value.tzinfo is not None and type(value.tzinfo) not in (datetime.timezone, ZoneInfo):
            raise ResultError([issue(path, value, 'date/time with builtin timezone', 'Custom timezone conversion is unavailable')])
        value = value.isoformat()
    if type(value) not in (str, int, float, bool, type(None)):
        raise ResultError([issue(path, original, 'JSON scalar: string, finite number, boolean or null',
                                 'Metrics must contain JSON scalar values; arrays, nested structures and custom objects are unavailable')])
    if type(value) is float and not math.isfinite(value):
        raise ResultError([issue(path, original, 'finite JSON numeric scalar', 'Non-finite numeric values are unavailable')])
    if type(value) is str and len(value) > string_limit:
        raise ResultError([issue(path, value, f'string up to {string_limit} characters', 'String exceeds the output contract')])
    return value


def canonicalize_result(result, np=None, pd=None, static=False, dataframe_type=()):
    if pd is not None:
        dataframe_type = pd.DataFrame
    if static and result is UNKNOWN:
        return result
    if type(result) is not dict:
        raise ResultError([issue('result', result, 'canonical envelope or flat scalar dictionary',
                                 'Set result to a canonical envelope or flat metrics dictionary')])
    if len(result) > METRIC_LIMIT:
        raise ResultError([issue('result', result, 'bounded result dictionary', 'Result has too many fields')])
    if any(type(key) is not str for key in result):
        raise ResultError([issue('result', result, 'string keys', 'Result keys must be strings')])
    reserved = RESULT_SCHEMA['properties']
    if any(key in reserved for key in result):
        missing = [key for key in RESULT_SCHEMA['required'] if key not in result]
        unexpected = [key for key in result if key not in reserved]
        if missing or unexpected:
            raise ResultError([issue('result', result, 'canonical summary/metrics/tables envelope',
                'Do not mix reserved envelope keys with flat metrics', missing=missing, unexpected=unexpected)])
        summary, metrics, tables = result.get('summary', ''), result.get('metrics'), result.get('tables', {})
    else:
        summary, metrics, tables = '', result, {}
    errors = []
    try:
        summary = scalar(summary, 'result.summary', np, pd, static, SUMMARY_LIMIT)
        if not (static and summary is UNKNOWN) and type(summary) is not str:
            errors.append(issue('result.summary', summary, 'bounded string', 'Result summary exceeds the output contract'))
    except ResultError as error:
        errors.extend(error.issues)
    normalized = UNKNOWN if static and metrics is UNKNOWN else {}
    if not (static and metrics is UNKNOWN):
        if type(metrics) is not dict or len(metrics) > METRIC_LIMIT:
            errors.append(issue('result.metrics', metrics, 'at most 32 named scalar values', 'Metrics must be at most 32 named scalar values'))
        else:
            for key, value in metrics.items():
                if type(key) is not str or len(key) > METRIC_NAME_LIMIT:
                    errors.append(issue('result.metrics', key, 'string key up to 80 characters', 'Invalid metric name'))
                    continue
                try:
                    normalized[key] = scalar(value, 'result.metrics.' + key, np, pd, static)
                except ResultError as error:
                    errors.extend(error.issues)
    if not (static and tables is UNKNOWN):
        if type(tables) is not dict or len(tables) > TABLE_LIMIT:
            errors.append(issue('result.tables', tables, 'at most eight named DataFrames', 'Tables must be an object with at most eight named tables'))
        else:
            for name, table in tables.items():
                path = 'result.tables.' + name if type(name) is str else 'result.tables'
                if type(name) is not str or not name.replace('_', '').isalnum() or len(name) > TABLE_NAME_LIMIT:
                    errors.append(issue(path, table, 'safe table name', 'Unsafe table artifact name'))
                elif not (static and table is UNKNOWN) and type(table) is not dataframe_type:
                    errors.append(issue(path, table, 'pandas.DataFrame', 'Named tables must be pandas DataFrames; large lists are not an output format'))
    if errors:
        raise ResultError(errors)
    return {'summary': summary, 'metrics': normalized, 'tables': dict(tables) if type(tables) is dict else tables}


def result_issues(result, dataframe_type):
    """Only prove literal incompatibilities on the host; runtime checks dynamic values."""
    try:
        canonicalize_result(result, static=True, dataframe_type=dataframe_type)
    except ResultError as error:
        return error.issues
    return []
