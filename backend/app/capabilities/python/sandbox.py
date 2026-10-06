"""Bubblewrap execution with explicit inputs, bounded output and no unsafe fallback."""
import asyncio
import base64
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import signal
import sys
import tempfile

TIMEOUT_SECONDS = 15
ARTIFACT_LIMIT = 16
ARTIFACT_BYTES = 8 * 1024 * 1024
TOTAL_ARTIFACT_BYTES = 32 * 1024 * 1024


async def bounded_read(stream, limit):
    data = bytearray()
    while chunk := await stream.read(64 * 1024):
        if len(data) + len(chunk) > limit:
            raise ValueError('Sandbox transfer exceeds the output bound')
        data.extend(chunk)
    return bytes(data)


def artifact_root():
    return Path(os.getenv('TASKFLOW_ARTIFACTS_DIR', Path(__file__).resolve().parents[4] / 'data' / 'artifacts')).resolve()


async def isolated_stage(code, input_paths, destination, stage, timeout=TIMEOUT_SECONDS):
    executable = shutil.which('bwrap')
    if not executable:
        return {'ok': False, 'error': 'Sandbox unavailable: install bubblewrap; unsafe execution is disabled', 'error_code': 'SANDBOX_UNAVAILABLE', 'category': 'SANDBOX_UNAVAILABLE'}
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='taskflow-job-') as temporary:
        job = Path(temporary)
        input_dir = job / 'inputs'
        input_dir.mkdir()
        mapped = {}
        input_hashes = {}
        for index, (document_id, source) in enumerate(input_paths.items()):
            filename = f'input_{index}.csv'
            shutil.copyfile(source, input_dir / filename)
            mapped[document_id] = '/input/' + filename
            input_hashes[document_id] = hashlib.sha256((input_dir / filename).read_bytes()).hexdigest()
        (job / 'analysis.py').write_text(code)
        (job / 'spec.json').write_text(json.dumps({'inputs': mapped, 'stage': stage}))
        command = [executable, '--unshare-all', '--die-with-parent', '--new-session', '--cap-drop', 'ALL',
                   '--ro-bind', '/usr', '/usr', '--symlink', 'usr/lib', '/lib', '--symlink', 'usr/lib64', '/lib64',
                   '--ro-bind', str(Path(sys.prefix).resolve()), '/opt/venv', '--proc', '/proc', '--remount-ro', '/proc', '--dev', '/dev', '--remount-ro', '/dev',
                   '--ro-bind', str(input_dir), '/input', '--ro-bind', str(job / 'analysis.py'), '/job/analysis.py',
                   '--ro-bind', str(job / 'spec.json'), '/job/spec.json',
                   '--ro-bind', str(Path(__file__).with_name('worker.py').resolve()), '/job/worker.py',
                   '--ro-bind', str(Path(__file__).with_name('contracts.py').resolve()), '/job/contracts.py',
                   '--size', str(TOTAL_ARTIFACT_BYTES), '--tmpfs', '/output', '--dir', '/tmp', '--remount-ro', '/', '--chdir', '/output', '--clearenv']
        for key, value in {'PATH': '/opt/venv/bin:/usr/bin', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                           'MKL_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': '0', 'HOME': '/nonexistent'}.items():
            command += ['--setenv', key, value]
        command += ['/opt/venv/bin/python', '-B', '-I', '/job/worker.py']
        process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
        readers = [asyncio.create_task(bounded_read(process.stdout, 48 * 1024 * 1024)), asyncio.create_task(bounded_read(process.stderr, 8192))]
        async def collect_output():
            streams = await asyncio.gather(*readers)
            await process.wait()
            return streams
        try:
            stdout, stderr = await asyncio.wait_for(collect_output(), timeout)
        except (TimeoutError, ValueError, asyncio.CancelledError) as error:
            for reader in readers:
                reader.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            await asyncio.wait_for(process.communicate(), 2)
            if isinstance(error, asyncio.CancelledError):
                raise
            if isinstance(error, ValueError):
                return {'ok': False, 'error': str(error), 'error_code': 'OUTPUT_LIMIT', 'category': 'PYTHON_OUTPUT_LIMIT'}
            return {'ok': False, 'error': 'Sandbox execution deadline exceeded', 'error_code': 'PYTHON_TIMEOUT', 'category': 'PYTHON_TIMEOUT'}
        if process.returncode:
            category = 'PYTHON_TIMEOUT' if process.returncode == -signal.SIGXCPU else 'SANDBOX_UNAVAILABLE' if b'bwrap:' in stderr else 'PYTHON_RUNTIME_ERROR'
            return {'ok': False, 'error': ('Sandbox worker failed: ' + stderr.decode(errors='replace'))[:2000], 'error_code': 'SANDBOX_WORKER_FAILED', 'category': category}
        try:
            result = json.loads(stdout)
            if not isinstance(result, dict) or not isinstance(result.get('ok'), bool):
                raise ValueError('Invalid worker result')
            payloads = result.pop('artifact_payloads', [])
            if len(payloads) > ARTIFACT_LIMIT or len(json.dumps(result).encode()) > 80 * 1024:
                raise ValueError('Worker output exceeds the supported bound')
            total = 0
            for payload in payloads:
                name = payload['name']
                if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', name) or name == 'result.json':
                    raise ValueError('Invalid artifact filename')
                content = base64.b64decode(payload['base64'], validate=True)
                total += len(content)
                if len(content) > ARTIFACT_BYTES or total > TOTAL_ARTIFACT_BYTES:
                    raise ValueError('Artifact exceeds the output bound')
                (destination / name).write_bytes(content)
        except (ValueError, KeyError, TypeError):
            return {'ok': False, 'error': 'Invalid or oversized sandbox output', 'error_code': 'OUTPUT_LIMIT', 'category': 'PYTHON_OUTPUT_LIMIT'}
        result['stage'] = stage
        result['input_hashes'] = input_hashes
        return result


def collect_artifacts(directory, run_id):
    artifacts = []
    total = 0
    for path in sorted(directory.rglob('*')):
        if path.is_symlink() or not path.is_file() or path.parent != directory:
            raise ValueError('Artifacts must be regular files directly inside the output directory')
        if path.name == 'result.json':
            continue
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', path.name):
            raise ValueError('Artifact names must be plain filenames')
        total += path.stat().st_size
        if path.stat().st_size > ARTIFACT_BYTES or total > TOTAL_ARTIFACT_BYTES or len(artifacts) >= ARTIFACT_LIMIT:
            raise ValueError('Artifact count or size exceeds the output bound')
        artifacts.append({'name': path.name, 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'reference': f'/api/runs/{run_id}/artifacts/{directory.name}/{path.name}'})
    return artifacts
