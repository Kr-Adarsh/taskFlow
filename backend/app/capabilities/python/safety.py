"""Early code rejection; OS isolation remains the security boundary."""
import ast
from pathlib import PurePath
from backend.app.capabilities.python.contracts import IMPORTS, UNKNOWN, result_issues

FORBIDDEN_NAMES = {'eval', 'exec', 'compile', 'open', 'input', 'globals', 'locals', 'vars', 'getattr', 'setattr', 'delattr', '__import__', 'breakpoint', 'help'}
FORBIDDEN_ATTRIBUTES = {'read_pickle', 'to_pickle', 'read_sql', 'read_sql_query', 'read_sql_table', 'read_html', 'read_xml', 'read_clipboard', 'to_clipboard', 'load', 'loads', 'save', 'savez', 'ctypes', 'system', 'popen', 'spawn', 'fork', 'socket', 'connect'}


def code_issues(code):
    if not isinstance(code, str) or not code.strip() or len(code) > 12000:
        return None, [{'type': 'invalid_code', 'message': 'Code must contain 1–12000 characters'}]
    try:
        tree = ast.parse(code, filename='analysis.py')
    except SyntaxError as error:
        return None, [{'type': 'syntax_error', 'line': error.lineno, 'message': error.msg}]
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in IMPORTS:
                    issues.append({'type': 'unsupported_import', 'module': alias.name,
                                   'message': 'Import is outside the supported analysis libraries'})
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in IMPORTS or any(alias.name == '*' or alias.name.startswith('_') for alias in node.names):
                issues.append({'type': 'unsupported_import', 'module': node.module, 'level': node.level,
                               'message': 'Import is outside the supported analysis libraries'})
        elif isinstance(node, ast.Name) and (node.id.startswith('_') or node.id in FORBIDDEN_NAMES):
            issues.append({'type': 'unavailable_name', 'name': node.id, 'message': f'Name is unavailable: {node.id}'})
        elif isinstance(node, ast.Attribute) and (node.attr.startswith('_') or node.attr in FORBIDDEN_ATTRIBUTES):
            issues.append({'type': 'unavailable_attribute', 'attribute': node.attr, 'message': f'Attribute is unavailable: {node.attr}'})
    return tree, issues


def validate_code(code):
    tree, issues = code_issues(code)
    if issues:
        raise ValueError(issues[0]['message'])
    return tree


def static_value(node):
    if isinstance(node, ast.Dict):
        values = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                return UNKNOWN
            try:
                name = ast.literal_eval(key)
                hash(name)
            except (ValueError, TypeError):
                return UNKNOWN
            values[name] = static_value(value)
        return values
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return UNKNOWN


def static_result(tree):
    # Later references may mutate a literal result. Branches/aliases/calls defer to the worker.
    for statement in reversed(tree.body):
        references = [node for node in ast.walk(statement) if isinstance(node, ast.Name) and node.id == 'result']
        if not references:
            if any(isinstance(node, ast.Call) for node in ast.walk(statement)):
                return UNKNOWN
            continue
        if isinstance(statement, ast.Assign) and all(isinstance(target, ast.Name) for target in statement.targets):
            if any(target.id == 'result' for target in statement.targets):
                return static_value(statement.value)
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name) and statement.target.id == 'result' and statement.value:
            return static_value(statement.value)
        return UNKNOWN
    return UNKNOWN


def program_issues(code, input_names):
    tree, issues = code_issues(code)
    if tree is None:
        return issues
    readers = {'pd'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            readers.update(alias.asname or alias.name for alias in node.names if alias.name == 'pandas')
    methods = {'read_csv', 'read_parquet', 'read_json', 'read_excel'}
    imported_readers = {alias.asname or alias.name for node in ast.walk(tree)
                        if isinstance(node, ast.ImportFrom) and node.module == 'pandas'
                        for alias in node.names if alias.name in methods}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        reader = (isinstance(node.func, ast.Attribute) and node.func.attr in methods
                  and isinstance(node.func.value, ast.Name) and node.func.value.id in readers)
        reader = reader or isinstance(node.func, ast.Name) and node.func.id in imported_readers | {'open'}
        if not reader:
            continue
        path = node.args[0] if node.args else next((kw.value for kw in node.keywords
            if kw.arg in {'filepath_or_buffer', 'path_or_buf', 'path', 'io', 'file'}), None)
        if not isinstance(path, ast.Constant) or not isinstance(path.value, str):
            continue
        for name in input_names:
            if path.value == name or PurePath(path.value).name == PurePath(name).name:
                issues.append({'type': 'invalid_input_access', 'input': name, 'path': path.value,
                               'expected': f'inputs[{name!r}]', 'message': 'Read supplied inputs through the inputs mapping'})
    issues.extend(result_issues(static_result(tree), dataframe_type=()))
    return issues
