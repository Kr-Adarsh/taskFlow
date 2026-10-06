"""Independent, read-only checks for each capability and original-goal coverage."""
import json
import re
from backend.app.agent.prompts import DATA_BOUNDARY
from backend.app.agent.verifier import VerifierEngine, snapshot_state, read_sources, state_delta
from backend.app.agent_v2.state import Verification, AnswerAssessment, CoverageAssessment
from backend.app.capabilities.documents import chunks


def normalize_quote(text):
    return re.sub(r'\s+', ' ', text).strip()


def no_mutations(before, after):
    return not any(rows for changes in state_delta(before, after).values() for rows in changes.values())


class CapabilityVerifier:
    def __init__(self, provider, db_path=None):
        self.provider = provider
        self.db_path = db_path
        self.browser = VerifierEngine(db_path=db_path, provider=provider)

    async def verify(self, objective, task, pre_state, memory, python_result=None):
        after = snapshot_state(self.db_path)
        delta = state_delta(pre_state, after)
        capability = task.verification_capability
        if capability in ('documents', 'python') and not no_mutations(pre_state, after):
            return Verification(outcome='FATAL_FAILURE', verified=False,
                summary='A read-only capability caused unexpected business-state mutations.',
                evidence={'state_delta': delta}, discrepancies=['Read-only task changed application state'])
        if capability == 'browser':
            result = await self.browser.verify_run(objective, memory.get_snapshot(), success_criteria=task.success_criteria,
                pre_state=pre_state, post_state=after, source_evidence=read_sources(self.db_path), reported_result=task.result)
            failure = result.context.get('interpretation_failure') or {}
            outcome = 'PASS' if result.verified else failure.get('outcome', 'RECOVERABLE_FAILURE')
            return Verification(outcome=outcome, verified=result.verified,
                                summary=result.summary, discrepancies=result.discrepancies,
                                evidence={'verification_result': result.model_dump(), 'state_delta': delta})
        if capability == 'documents':
            return await self.verify_answer(objective, task, pre_state, after)
        if capability == 'python':
            return await self.verify_computation(objective, task, pre_state, after, memory, python_result)
        return Verification(outcome='FATAL_FAILURE', verified=False, summary='Unsupported verification capability.')

    async def verify_answer(self, objective, task, pre_state, after):
        references = []
        try:
            for reference in task.evidence:
                found = next((chunk for chunk in chunks(reference.document_id) if chunk['chunk_id'] == reference.chunk_id), None)
                if not found or not reference.quote.strip() or normalize_quote(reference.quote) not in normalize_quote(found['text']):
                    raise ValueError('Source quote does not match its identified chunk')
                references.append({**reference.model_dump(), 'source_text': found['text']})
            if not task.result or not references:
                raise ValueError('Material answer claims require valid source quotes')
        except (ValueError, OSError) as error:
            return Verification(outcome='AMBIGUITY', verified=False, summary=str(error))
        assessment, _ = await self.provider.generate_structured([
            {'role': 'system', 'content': DATA_BOUNDARY + '\nIndependently compare the original question, answer and re-read source evidence. List material claims; reject unsupported identities, relative selections, conditions or computed facts. All original requirements and every material claim must be covered. Valid quotes alone are insufficient. JSON schema: ' + json.dumps(AnswerAssessment.model_json_schema())},
            {'role': 'user', 'content': json.dumps({'original_objective': objective, 'answer': task.result, 'untrusted_sources': references})},
        ], AnswerAssessment)
        passed = assessment.covered and bool(assessment.material_claims) and not assessment.unsupported_claims and no_mutations(pre_state, after)
        return Verification(outcome='PASS' if passed else 'RECOVERABLE_FAILURE', verified=passed,
            summary=assessment.reason, evidence={'references': references, 'assessment': assessment.model_dump(), 'state_delta': state_delta(pre_state, after)},
            discrepancies=assessment.unsupported_claims)

    async def verify_computation(self, objective, task, pre_state, after, memory, python_result):
        import hashlib
        import math
        from backend.app.capabilities.python.verification import CalculationContract, calculate
        from backend.app.capabilities.python.profile import context_profile
        from backend.app.capabilities.documents import resolve_file
        from backend.app.capabilities.python.sandbox import artifact_root
        if not python_result or not python_result.get('ok') or python_result.get('stage') != 'full':
            return Verification(outcome='RECOVERABLE_FAILURE', verified=False, summary='A successful full-dataset execution is required.')
        profiles = memory.get_snapshot()['dataset_profiles']
        try:
            for source in python_result['inputs']:
                _, path = resolve_file(source['document_id'])
                if source['sha256'] != hashlib.sha256(path.read_bytes()).hexdigest() or source['document_id'] not in profiles:
                    raise ValueError('Analysis input provenance changed or is missing')
            for artifact in python_result.get('artifacts', []):
                parts = artifact['reference'].split('/')
                if len(parts) != 7 or parts[1:3] != ['api', 'runs'] or parts[4] != 'artifacts' or parts[3] != python_result.get('run_id'):
                    raise ValueError('Invalid artifact reference')
                path = artifact_root() / parts[3] / parts[5] / parts[6]
                if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(artifact_root()) or hashlib.sha256(path.read_bytes()).hexdigest() != artifact['sha256']:
                    raise ValueError('Expected artifact is missing or changed')
            actual = python_result['metrics']
            if task.result.get('metrics') != actual:
                raise ValueError('Final answer metrics differ from the full execution result')
            contract, _ = await self.provider.generate_structured([
                {'role': 'system', 'content': DATA_BOUNDARY + '\nIndependently specify a deterministic grouped numeric aggregation or period comparison from the ORIGINAL objective and dataset schema. Never use executor values as expected answers. Preserve sum/mean/count/min/max, absolute versus percent change, baseline/current periods, direction and requested min/max selection. Choose metric keys from the output schema only. All original requirements must fit this contract; otherwise list unsupported_requirements. JSON schema: ' + json.dumps(CalculationContract.model_json_schema())},
                {'role': 'user', 'content': json.dumps({'original_objective': objective, 'dataset_profiles': {key: context_profile(profile, objective) for key, profile in profiles.items()}, 'metric_keys': list(actual)})},
            ], CalculationContract)
            if contract.unsupported_requirements:
                raise ValueError('Unsupported calculation requirements: ' + '; '.join(contract.unsupported_requirements))
            if contract.document_id not in {source['document_id'] for source in python_result['inputs']}:
                raise ValueError('Verification input was not used by the computation')
            expected = calculate(contract)
            value = actual.get(contract.value_metric)
            passed = (not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
                      and math.isclose(value, expected['value'], rel_tol=1e-9, abs_tol=1e-6)
                      and str(actual.get(contract.group_metric)) == expected['group'] and no_mutations(pre_state, after))
            return Verification(outcome='PASS' if passed else 'RECOVERABLE_FAILURE', verified=passed,
                summary='Full-data calculation and output artifacts independently checked.' if passed else 'Computed result differs from the independent full-data calculation.',
                evidence={'contract': contract.model_dump(), 'expected': expected, 'actual': actual, 'artifacts': python_result.get('artifacts', []), 'state_delta': state_delta(pre_state, after)},
                discrepancies=[] if passed else ['The selected group or calculated measure does not match an independent full-data check; review aggregation, periods, units and direction.'])
        except (ValueError, KeyError, OSError) as error:
            return Verification(outcome='AMBIGUITY', verified=False, summary=str(error))

    @staticmethod
    def audit_mutations(tasks, before, after):
        deltas = [task.result['verification']['evidence'].get('state_delta') for task in tasks]
        actual_delta = state_delta(before, after)
        if not deltas or any(delta is None for delta in deltas):
            return Verification(outcome='FATAL_FAILURE', verified=False, summary='A task has no independent mutation audit.')
        if len(tasks) == 1:
            if actual_delta != deltas[0]:
                return Verification(outcome='FATAL_FAILURE', verified=False,
                    summary='The final workspace differs from the independently verified state.', evidence={'state_delta': actual_delta})
        else:
            for collection, changes in actual_delta.items():
                for kind, rows in changes.items():
                    approved = [row for delta in deltas for row in delta[collection][kind]]
                    if any(row not in approved for row in rows):
                        return Verification(outcome='FATAL_FAILURE', verified=False,
                            summary='The final workspace contains changes outside verified task outcomes.', evidence={'state_delta': actual_delta})
        return Verification(outcome='PASS', verified=True, summary='Final mutations match independently verified task outcomes.',
                            evidence={'state_delta': actual_delta})

    async def verify_coverage(self, objective, tasks, before, after):
        audit = self.audit_mutations(tasks, before, after)
        if not audit.verified:
            return audit
        actual_delta = audit.evidence['state_delta']
        proofs = []
        for task in tasks:
            proofs.append({'task_id': task.task_id, 'goal': task.goal, 'status': task.status.value,
                           'result': {key: value for key, value in task.result.items() if key != 'verification'},
                           'independent_verification': task.result['verification']})
        assessment, _ = await self.provider.generate_structured([
            {'role': 'system', 'content': DATA_BOUNDARY + '\nCheck whether the separately verified subtask outcomes collectively cover every requirement and condition of the ORIGINAL request. Do not accept a weakened plan or an executor claim without its independent verification. Check that all persisted changes belong to verified tasks. Missing coverage must remain unresolved. JSON schema: ' + json.dumps(CoverageAssessment.model_json_schema())},
            {'role': 'user', 'content': json.dumps({'original_objective': objective, 'verified_tasks': proofs, 'state_delta': actual_delta})},
        ], CoverageAssessment)
        passed = assessment.covered and not assessment.missing_requirements
        return Verification(outcome='PASS' if passed else 'FATAL_FAILURE', verified=passed, summary=assessment.reason,
                            discrepancies=assessment.missing_requirements, evidence={'coverage': assessment.model_dump()})
