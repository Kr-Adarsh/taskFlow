"""Reusable dataset profiling and bounded model-generated local analysis."""
import hashlib
import shutil
from pathlib import Path
import tempfile
import uuid

from backend.app.tools.base import ToolResult
from backend.app.workspace.lease import mutation_owner
from backend.app.capabilities.python.profile import dataset_profile, load_dataset
from backend.app.capabilities.python.safety import program_issues
from backend.app.capabilities.python.contracts import IMPORTS, RESULT_CONTRACT, MAX_ATTEMPTS
from backend.app.capabilities.python.sandbox import artifact_root, isolated_stage, collect_artifacts


class PythonCapability:
    def __init__(self, run_id=None):
        self.run_id = run_id
        self.profiles = {}
        self.attempts = {}
        self.failed_programs = {}
        self.task_id = 'default'
        self.calls = 0

    def profile_dataset(self, document_id):
        try:
            profile = dataset_profile(document_id)
            self.profiles[document_id] = profile
            return ToolResult(ok=True, data=profile, evidence={'document_id': document_id, 'sha256': profile['sha256']})
        except (ValueError, OSError) as error:
            return ToolResult(ok=False, error=str(error)[:1200], error_code='DATASET_UNAVAILABLE')

    async def execute_python(self, document_ids, code):
        owner = mutation_owner.get()
        run_id = self.run_id or (owner[0] if owner else None)
        if not run_id:
            return ToolResult(ok=False, error='Analysis requires an owned run', error_code='RUN_REQUIRED')
        scope = (run_id, self.task_id)
        attempt = self.attempts.get(scope, 0)
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        remaining = max(0, MAX_ATTEMPTS - attempt)
        budget = {'attempt_number': attempt, 'attempts_remaining': remaining,
                  'repair_remaining': remaining > 0, 'repair_attempts_remaining': remaining}
        if code_hash in self.failed_programs.get(scope, {}):
            data = {**budget, 'category': self.failed_programs[scope][code_hash], 'no_progress': True,
                    'no_op': True, 'code_sha256': code_hash}
            return ToolResult(ok=False, error='This exact program already failed; select a different program',
                              error_code='REDUNDANT_ACTION', data=data, evidence=data)
        if remaining == 0:
            return ToolResult(ok=False, error='Python task exhausted its three program attempts',
                              error_code='CODE_REPAIR_EXHAUSTED', data=budget, evidence=budget)
        attempt += 1
        self.attempts[scope] = attempt
        self.calls += 1
        budget.update(attempt_number=attempt, attempts_remaining=MAX_ATTEMPTS - attempt,
                      repair_remaining=attempt < MAX_ATTEMPTS, repair_attempts_remaining=MAX_ATTEMPTS - attempt)
        destination = artifact_root() / run_id / ('analysis_' + uuid.uuid4().hex[:12])
        result = None
        try:
            issues = program_issues(code, document_ids)
            if issues:
                security = any(issue['type'] in ('unsupported_import', 'unavailable_name', 'unavailable_attribute') for issue in issues)
                self.failed_programs.setdefault(scope, {})[code_hash] = 'PYTHON_SECURITY_DENIED' if security else 'PYTHON_CONTRACT_ERROR'
                contract_error = {
                    **budget, 'category': 'PYTHON_SECURITY_DENIED' if security else 'PYTHON_CONTRACT_ERROR',
                    'issues': issues,
                    'python_contract': {'available_inputs': document_ids, 'input_access': 'inputs[<name>]',
                        'allowed_imports': sorted(IMPORTS), 'result_contract': RESULT_CONTRACT,
                        'scaffold': 'df = pd.read_csv(inputs["<registered input name>"])\n# analysis\nresult = {"summary": "...", "metrics": {}, "tables": {}}'},
                }
                return ToolResult(ok=False, error='Generated Python violates the execution contract',
                                  error_code='PYTHON_CONTRACT_ERROR', data=contract_error, evidence=contract_error)
            datasets = {}
            for document_id in document_ids:
                if document_id not in self.profiles:
                    raise ValueError('Profile every dataset before generating code')
                path, frame = load_dataset(document_id)
                if hashlib.sha256(path.read_bytes()).hexdigest() != self.profiles[document_id]['sha256']:
                    raise ValueError('Input changed after profiling; profile it again')
                datasets[document_id] = (path, frame)
            if not datasets:
                raise ValueError('At least one profiled CSV input is required')
            with tempfile.TemporaryDirectory(prefix='taskflow-sample-') as temporary:
                sample_paths = {}
                for index, (document_id, (_, frame)) in enumerate(datasets.items()):
                    sample = frame.iloc[sorted({round(i * (len(frame) - 1) / 19) for i in range(20)})]
                    path = Path(temporary) / f'sample_{index}.csv'; sample.to_csv(path, index=False)
                    sample_paths[document_id] = path
                result = await isolated_stage(code, sample_paths, Path(temporary) / 'output', 'sample')
            if result['ok']:
                result = await isolated_stage(code, {key: value[0] for key, value in datasets.items()}, destination, 'full')
            if result['ok']:
                if any(result['input_hashes'].get(document_id) != self.profiles[document_id]['sha256'] for document_id in datasets):
                    raise ValueError('Copied execution input differs from the profiled source')
                result['artifacts'] = collect_artifacts(destination, run_id)
                result['inputs'] = [{'document_id': document_id, 'sha256': self.profiles[document_id]['sha256'], 'rows': len(frame)} for document_id, (_, frame) in datasets.items()]
                result['code_sha256'] = hashlib.sha256(code.encode()).hexdigest()
                result['run_id'] = run_id
                result.update(budget)
                return ToolResult(ok=True, data=result, evidence={'run_id': run_id, 'code_sha256': result['code_sha256'], 'stage': 'full', **budget})
        except (ValueError, SyntaxError, OSError) as error:
            result = {'ok': False, 'error': f'{type(error).__name__}: {str(error)[:1200]}', 'error_code': 'CODE_REJECTED', 'category': 'PYTHON_CONTRACT_ERROR'}
        self.failed_programs.setdefault(scope, {})[code_hash] = result.get('category', result.get('error_code', 'PYTHON_RUNTIME_ERROR'))
        result.update(budget)
        shutil.rmtree(destination, ignore_errors=True)
        return ToolResult(ok=False, data=result, error=result['error'], error_code=result.get('error_code', 'PYTHON_RUNTIME_ERROR'),
                          evidence=budget)



def register_python(registry):
    from backend.app.capabilities.registry import schema
    capability = PythonCapability()
    registry.python = capability
    registry.add('python', 'profile_dataset', 'Profile a registered CSV locally: shape, columns, dtypes, three sample rows, null/unique counts and numeric/categorical summaries. Full data stays local.', schema(document_id={'type': 'string'}), capability.profile_dataset)
    registry.add('python', 'execute_python', 'Execute analysis after profiling. Inputs are read-only CSV paths in inputs[document_id]; pd, np, math, json are available. Set result={"summary":str,"metrics":dict,"tables":{name:DataFrame}}; tables become artifacts with three-row previews. Write files only under output_dir. Sample validation precedes FULL dataset execution. Supported imports: pandas,numpy,math,statistics,datetime,json. No network, shell, subprocess, private attributes or arbitrary file access. A flat dictionary of named scalar metrics is also accepted; do not mix envelope keys with flat metrics. Safe NumPy scalars and dates normalize to JSON values. Maximum three distinct program attempts per task; identical failed code is no progress. Errors expose attempt_number and attempts_remaining. No bulk outputs.',
                 schema(document_ids={'type':'array','items':{'type':'string'},'minItems':1,'maxItems':3,'uniqueItems':True},code={'type':'string','minLength':1,'maxLength':12000}), capability.execute_python)
