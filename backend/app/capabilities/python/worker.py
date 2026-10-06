"""Trusted worker mounted read-only inside a Linux namespace sandbox."""
import builtins
import base64
import contextlib
import ctypes
import errno
import io
import json
import math
import os
from pathlib import Path
import resource
import re
import sys
import traceback
import runpy

import numpy as np
import pandas as pd

MAX_BYTES = 64 * 1024
CONTRACT = runpy.run_path('/job/contracts.py')


class OutputLimitError(ValueError):
    pass


class BoundedStream(io.StringIO):
    def write(self, value):
        remaining = 4096 - self.tell()
        return super().write(value[:max(0, remaining)])


def restrict_syscalls():
    library = ctypes.CDLL('libseccomp.so.2', use_errno=True)
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    context = library.seccomp_init(0x7fff0000)
    if not context:
        raise RuntimeError('Cannot initialize syscall isolation')
    try:
        for name in ('execve', 'execveat', 'clone', 'clone3', 'fork', 'vfork', 'socket', 'socketpair', 'connect', 'bind', 'listen', 'accept', 'accept4', 'ptrace', 'mount', 'umount2', 'unshare', 'setns', 'bpf', 'userfaultfd'):
            number = library.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and library.seccomp_rule_add(context, 0x00050000 | errno.EPERM, number, 0) < 0:
                raise RuntimeError('Cannot configure syscall isolation')
        if library.seccomp_load(context) < 0:
            raise RuntimeError('Cannot load syscall isolation')
    finally:
        library.seccomp_release(context)


def audit(event, args):
    if event == 'open':
        filename, mode, flags = args
        if isinstance(filename, int):
            raise PermissionError('File descriptors are not available to generated code')
        path = Path(filename).resolve()
        writing = (isinstance(mode, str) and any(flag in mode for flag in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        allowed = path.is_relative_to('/output') if writing else any(path.is_relative_to(root) for root in ('/input', '/output', '/usr', '/opt/venv'))
        if not allowed:
            raise PermissionError('File access is outside the input/output contract')
    if event.startswith(('socket.', 'subprocess.', 'ctypes.')) or event in ('os.system', 'os.fork', 'os.posix_spawn', 'os.exec', 'os.symlink', 'os.link', 'os.chdir', 'sys.addaudithook'):
        raise PermissionError('Process, network or unsafe runtime access denied')


def run():
    spec = json.loads(Path('/job/spec.json').read_text())
    code = Path('/job/analysis.py').read_text()
    compiled = compile(code, 'analysis.py', 'exec')
    inputs = spec['inputs']
    resource.setrlimit(resource.RLIMIT_CPU, (8, 8))
    resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 ** 2, 8 * 1024 ** 2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    restrict_syscalls()
    safe_builtins = {key: value for key, value in vars(builtins).items() if not key.startswith('_') and key not in {'open', 'eval', 'exec', 'compile', 'input', 'globals', 'locals', 'vars', 'getattr', 'setattr', 'delattr', 'breakpoint', 'help'}}
    original_import = builtins.__import__
    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in CONTRACT['IMPORTS']:
            raise ImportError('Import is outside the supported analysis libraries')
        return original_import(name, globals, locals, fromlist, level)
    safe_builtins['__import__'] = safe_import
    namespace = {'__builtins__': safe_builtins, 'pd': pd, 'np': np, 'math': math, 'json': json, 'inputs': inputs, 'output_dir': '/output', 'stage': spec['stage']}
    stdout, stderr = BoundedStream(), BoundedStream()
    sys.addaudithook(audit)
    outcome = {'ok': False}
    phase = 'execute'
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compiled, namespace)
        phase = 'result'
        result = CONTRACT['canonicalize_result'](namespace.get('result'), np=np, pd=pd, dataframe_type=pd.DataFrame)
        tables = result.get('tables', {})
        for name, table in list(tables.items()):
            path = f'/output/{name}.csv'
            table.to_csv(path, index=False)
            tables[name] = {'shape': list(table.shape), 'columns': list(table.columns), 'preview': json.loads(table.head(3).to_json(orient='records')), 'artifact': name + '.csv'}
        encoded = json.dumps(result, allow_nan=False)
        if len(encoded.encode()) > MAX_BYTES:
            raise OutputLimitError('Structured output exceeds 64 KiB')
        outcome = {'ok': True, **result}
    except BaseException as error:
        category = ('PYTHON_SECURITY_DENIED' if isinstance(error, PermissionError) or isinstance(error, OSError) and error.errno in (errno.EPERM, errno.EACCES, errno.EROFS)
                    else 'PYTHON_OUTPUT_LIMIT' if isinstance(error, OutputLimitError) or isinstance(error, OSError) and error.errno == errno.EFBIG
                    else 'PYTHON_RESULT_ERROR' if phase == 'result' else 'PYTHON_RUNTIME_ERROR')
        outcome = {'ok': False, 'error': f'{type(error).__name__}: {str(error)[:1000]}',
                   'error_code': category, 'category': category, 'traceback': traceback.format_exc(limit=3)[-1800:]}
        if isinstance(error, CONTRACT['ResultError']):
            outcome['issues'] = error.issues
    outcome.update(stdout=stdout.getvalue(), stderr=stderr.getvalue(), isolation={'namespaces': True, 'seccomp': True})
    payloads = []
    total = 0
    try:
        if outcome['ok']:
            for path in sorted(Path('/output').iterdir()):
                if path.is_symlink() or not path.is_file() or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', path.name) or path.name == 'result.json':
                    raise ValueError('Artifacts must be regular files with plain filenames')
                size = path.stat().st_size
                total += size
                if len(payloads) >= 16 or size > 8 * 1024 ** 2 or total > 32 * 1024 ** 2:
                    raise OutputLimitError('Artifact count or size exceeds the output bound')
                payloads.append({'name': path.name, 'base64': base64.b64encode(path.read_bytes()).decode()})
    except (ValueError, OSError) as error:
        payloads = []
        category = 'PYTHON_OUTPUT_LIMIT' if isinstance(error, OutputLimitError) else 'PYTHON_RESULT_ERROR'
        outcome.update(ok=False, error=str(error)[:1000], error_code=category, category=category)
    outcome['artifact_payloads'] = payloads
    print(json.dumps(outcome, allow_nan=False))


if __name__ == '__main__':
    run()
