"""LangGraph orchestration around reusable, observable capabilities."""
import asyncio
from copy import deepcopy
import hashlib
import json
from jsonschema import Draft202012Validator
from langgraph.graph import StateGraph, START, END
from langsmith import tracing_context
from backend.app.agent.loop import AgentRunner as RunLifecycle
from backend.app.agent.schemas import AgentActionType
from backend.app.agent.verifier import snapshot_state
from backend.app.agent_v2.verifier import no_mutations
from backend.app.agent_v2.state import GraphState, Subtask, TaskPlan, TaskStatus, TaskSelection, Decision, Verification
from backend.app.agent_v2.planner import planner_prompt
from backend.app.agent_v2.context import ContextMemory, decision_prompt
from backend.app.agent_v2.verifier import CapabilityVerifier
from backend.app.agent_v2.oscillation import SemanticOscillationGuard
from backend.app.capabilities.registry import build_registry
from backend.app.tools.base import ToolResult
from backend.app.workspace.service import list_documents


def python_model_result_matches(reported, canonical):
    if not reported:
        return 'unknown'
    candidate = reported.get('metrics') if any(key in reported for key in ('summary', 'metrics', 'tables')) else reported
    if not isinstance(candidate, dict) or any(isinstance(value, (dict, list)) for value in candidate.values()):
        return 'unknown'
    try:
        json.dumps(reported, allow_nan=False)
    except (TypeError, ValueError):
        return 'unknown'
    actual = canonical['metrics']
    matches = candidate.keys() == actual.keys() and all(
        candidate[key] == actual[key] and isinstance(candidate[key], bool) == isinstance(actual[key], bool)
        for key in actual)
    return matches and all(reported[key] == canonical[key] for key in ('summary', 'tables') if key in reported)


class AgentRunner(RunLifecycle):
    def __init__(self, provider=None, tool_registry=None, verifier=None, **kwargs):
        super().__init__(provider=provider, tool_registry=tool_registry or build_registry(), **kwargs)
        self.capability_verifier = verifier or CapabilityVerifier(self.provider, self.db_path)
        self.oscillation_guard = SemanticOscillationGuard()
        self.oscillation_completed_tasks = frozenset()
        graph = StateGraph(GraphState)
        for name in ('intake', 'context_build', 'plan', 'select_subtask', 'select_capability', 'execute', 'observe', 'update_state', 'verify', 'finish'):
            graph.add_node(name, getattr(self, name))
        graph.add_edge(START, 'intake')
        graph.add_edge('intake', 'context_build')
        graph.add_conditional_edges('context_build', lambda state: state.route, {'plan': 'plan', 'select_subtask': 'select_subtask', 'select_capability': 'select_capability'})
        graph.add_edge('plan', 'select_subtask')
        graph.add_conditional_edges('select_subtask', lambda state: state.route, {'context_build': 'context_build', 'finish': 'finish'})
        graph.add_conditional_edges('select_capability', lambda state: state.route, {'execute': 'execute', 'verify': 'verify', 'plan': 'plan', 'finish': 'finish'})
        graph.add_edge('execute', 'observe')
        graph.add_edge('observe', 'update_state')
        graph.add_conditional_edges('update_state', lambda state: state.route, {'context_build': 'context_build', 'verify': 'verify'})
        graph.add_conditional_edges('verify', lambda state: state.route, {'select_subtask': 'select_subtask', 'context_build': 'context_build', 'finish': 'finish'})
        graph.add_edge('finish', END)
        self.graph = graph.compile()

    async def _execute_owned(self, objective, run_id):
        self.memory_mgr = ContextMemory()
        self.oscillation_guard = SemanticOscillationGuard()
        self.oscillation_completed_tasks = frozenset()
        self.source_versions = {}
        self.python_results = {}
        self.task_pre_states = {}
        self.mutation_checkpoint_states = {}
        self.last_tasks = []
        self.file_catalog = [{'document_id': doc.filename, 'title': doc.title, 'type': doc.doc_type} for doc in list_documents(self.db_path)][:30]
        with tracing_context(enabled=False):
            state = await self.graph.ainvoke(GraphState(run_id=run_id, objective=objective), config={'recursion_limit': self.max_steps * 10 + 100, 'callbacks': []})
        return state['terminal']

    def current(self, state):
        return next(task for task in state.tasks if task.task_id == state.current_task_id)

    def publish_tasks(self, state, tasks):
        self.last_tasks = tasks
        self._emit_event(state.run_id, 'TASK_GRAPH', {'tasks': [task.model_dump() for task in tasks]})
        self._update_run_status(state.run_id, 'running')

    async def intake(self, state):
        self.pre_state = snapshot_state(self.db_path)
        self._emit_event(state.run_id, 'TASK', {'objective': state.objective, 'runtime': 'v2'})
        return {'observation': {'data': {'message': 'Workspace apps: /workspace/finance, /workspace/crm, /workspace/support, /workspace/documents.', 'file_catalog': self.file_catalog}}}

    async def context_build(self, state):
        return {'route': 'plan' if not state.tasks else 'select_capability' if state.current_task_id else 'select_subtask'}

    async def plan(self, state):
        completed = [task for task in state.tasks if task.status == TaskStatus.COMPLETED]
        plan, metadata = await self.provider.generate_structured(planner_prompt(state.objective, [task.model_dump() for task in completed]), TaskPlan)
        by_id = {task.task_id: task for task in completed}
        tasks = []
        for task in plan.tasks:
            if task.task_id in by_id:
                old = by_id.pop(task.task_id)
                if old.goal != task.goal:
                    raise ValueError('Replan changed a completed task goal')
                tasks.append(old)
            else:
                tasks.append(Subtask(**task.model_dump()))
        if by_id:
            raise ValueError('Replan omitted completed tasks')
        self._emit_event(state.run_id, 'MODEL_CALL', {'stage': 'planner', 'metadata': metadata})
        self._emit_event(state.run_id, 'PLAN', plan.model_dump())
        self._update_run_status(state.run_id, 'running', plan=plan.model_dump())
        self.publish_tasks(state, tasks)
        return {'tasks': tasks, 'success_criteria': plan.success_criteria, 'current_task_id': None}

    async def select_subtask(self, state):
        if state.terminal:
            return {'route': 'finish'}
        complete = {task.task_id for task in state.tasks if task.status == TaskStatus.COMPLETED}
        ready = [task for task in state.tasks if task.status == TaskStatus.PENDING and set(task.dependencies) <= complete]
        if not ready:
            return {'route': 'finish', 'terminal': None if len(complete) == len(state.tasks) else {'run_id': state.run_id, 'status': 'failed', 'error': 'Task graph has blocked dependencies'}}
        selected = ready[0]
        if len(ready) > 1:
            selection, meta = await self.provider.generate_structured([
                {'role': 'system', 'content': 'Choose the next dependency-ready subtask for the original objective. JSON schema: ' + json.dumps(TaskSelection.model_json_schema())},
                {'role': 'user', 'content': json.dumps({'original_objective': state.objective, 'ready_tasks': [task.model_dump() for task in ready]})}], TaskSelection)
            selected = next((task for task in ready if task.task_id == selection.task_id), None)
            if selected is None:
                raise ValueError('Selected subtask is not dependency-ready')
            self._emit_event(state.run_id, 'MODEL_CALL', {'stage': 'subtask_selection', 'metadata': meta})
        selected.status = TaskStatus.RUNNING
        if complete != self.oscillation_completed_tasks:
            self.oscillation_guard = SemanticOscillationGuard()
            self.oscillation_completed_tasks = frozenset(complete)
        self.memory_mgr.no_progress_streak = 0
        self.task_pre_states.setdefault(selected.task_id, snapshot_state(self.db_path))
        self.publish_tasks(state, state.tasks)
        return {'tasks': state.tasks, 'current_task_id': selected.task_id, 'route': 'context_build'}

    async def select_capability(self, state):
        if state.terminal:
            return {'route': 'finish'}
        if self.memory_mgr.no_progress_streak >= self.memory_mgr.no_progress_limit:
            return {'route': 'finish', 'terminal': {'run_id': state.run_id, 'status': 'failed',
                    'error': 'No progress after four consecutive actions; repeated actions added no state or evidence'}}
        if state.steps >= self.max_steps:
            return {'route': 'finish', 'terminal': {'run_id': state.run_id, 'status': 'failed', 'error': 'V2 step budget exhausted'}}
        task = self.current(state)
        decision, metadata = await self.provider.generate_structured(decision_prompt(state, task, self.memory_mgr, self.registry.get_schemas()), Decision)
        self.step_count = state.steps + 1
        self._emit_event(state.run_id, 'DECISION', {'step': self.step_count, 'task_id': task.task_id, 'decision': decision.model_dump(), 'model_metadata': metadata})
        updates = {'decision': decision, 'steps': self.step_count}
        if decision.replan:
            if not no_mutations(self.task_pre_states[task.task_id], snapshot_state(self.db_path)):
                return {**updates, 'route': 'finish', 'terminal': {'run_id': state.run_id, 'status': 'failed', 'error': 'A changed workspace must be verified before replanning'}}
            if state.replans >= 1:
                return {**updates, 'route': 'finish', 'terminal': {'run_id': state.run_id, 'status': 'failed', 'error': 'Replan budget exhausted'}}
            return {**updates, 'route': 'plan', 'replans': state.replans + 1}
        if decision.action == AgentActionType.ACT:
            return {**updates, 'route': 'execute', 'capability': self.registry.categories.get(decision.tool_name, 'unknown')}
        if decision.action == AgentActionType.READY_FOR_VERIFICATION:
            if task.verification_capability == 'python':
                task.model_reported_result = deepcopy(decision.result)
                authoritative = task.authoritative_python_result
                execution = self.python_results.get(task.task_id)
                if (not authoritative or authoritative.get('task_id') != task.task_id
                    or authoritative.get('run_id') != state.run_id or authoritative.get('stage') != 'full'
                    or not execution or not execution.get('ok') or execution.get('stage') != 'full'):
                    error = ToolResult(ok=False, error='Python verification requires a successful full execution for this task',
                        error_code='PYTHON_RESULT_NOT_READY', data={'task_id': task.task_id, 'required_stage': 'full'})
                    task.status = TaskStatus.FAILED
                    task.errors.append({'step': self.step_count, 'code': error.error_code, 'error': error.error})
                    self._emit_event(state.run_id, 'OBSERVATION', {'step': self.step_count, 'task_id': task.task_id, **error.model_dump()})
                    return {**updates, 'tasks': state.tasks, 'route': 'finish', 'observation': error.model_dump(),
                        'terminal': {'run_id': state.run_id, 'status': 'failed', 'error': error.error, 'error_code': error.error_code}}
                task.result = deepcopy(authoritative['canonical_result'])
                task.evidence = []
                task.model_result_matches_authoritative = python_model_result_matches(decision.result, task.result)
                self._emit_event(state.run_id, 'PYTHON_RESULT_BINDING', {'task_id': task.task_id,
                    'model_reported_result': task.model_reported_result, 'authoritative_result': authoritative,
                    'model_result_matches_authoritative': task.model_result_matches_authoritative})
            else:
                task.result, task.evidence = decision.result, decision.evidence
            return {**updates, 'tasks': state.tasks, 'route': 'verify'}
        status = 'waiting_for_clarification' if decision.action == AgentActionType.NEED_CLARIFICATION else 'failed'
        task.status = TaskStatus.NEEDS_CLARIFICATION if status == 'waiting_for_clarification' else TaskStatus.FAILED
        return {**updates, 'tasks': state.tasks, 'route': 'finish', 'terminal': {'run_id': state.run_id, 'status': status, 'question' if status == 'waiting_for_clarification' else 'error': decision.clarification_question or decision.failure_reason}}

    async def execute(self, state):
        decision = state.decision
        warning = self.oscillation_guard.check_action(decision.tool_name, decision.tool_args,
            self.memory_mgr.get_snapshot(), snapshot_state(self.db_path))
        if warning:
            result = ToolResult(ok=False, error=warning['message'], error_code='OSCILLATION_DETECTED', data=warning)
            self._emit_event(state.run_id, 'OSCILLATION_DETECTED', {'step': state.steps, **warning})
            update = {'tool_result': result.model_dump()}
            if warning['fatal']:
                update['terminal'] = {'run_id': state.run_id, 'status': 'failed',
                    'error': 'Semantic oscillation repeated without new evidence or state after one warning',
                    'error_code': 'OSCILLATION_DETECTED'}
            return update
        if state.capability == 'python' and hasattr(self.registry, 'python'):
            self.registry.python.task_id = state.current_task_id
        self._emit_event(state.run_id, 'ACTION', {'step': state.steps, 'task_id': state.current_task_id, 'capability': state.capability,
                         'tool': decision.tool_name, 'args': decision.tool_args, 'thought': decision.thought})
        try:
            result = None
            if decision.tool_name == 'read_document_chunks':
                tool = self.registry.get(decision.tool_name)
                if tool and Draft202012Validator(tool.parameters).is_valid(decision.tool_args):
                    from backend.app.capabilities.documents import resolve_file
                    document_id = decision.tool_args['document_id']
                    version = None
                    try:
                        _, path = resolve_file(document_id)
                        stat = path.stat()
                        version = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                    except (ValueError, OSError):
                        pass
                    if version is not None and self.source_versions.get(document_id) == version:
                        result = self.memory_mgr.cached_document_read(decision.tool_args)
                    elif self.source_versions.get(document_id) != version:
                        self.memory_mgr.sources.pop(document_id, None)
                    self.source_versions[document_id] = version
            if result is None:
                result = await asyncio.wait_for(self.registry.execute(decision.tool_name, decision.tool_args), timeout=30)
        except TimeoutError:
            result = ToolResult(ok=False, error='Capability deadline exceeded', error_code='TOOL_TIMEOUT')
        return {'tool_result': result.model_dump()}

    async def observe(self, state):
        result = state.tool_result
        self._emit_event(state.run_id, 'OBSERVATION', {'step': state.steps, 'task_id': state.current_task_id, **result})
        return {'observation': result}

    async def update_state(self, state):
        result = ToolResult.model_validate(state.tool_result)
        task = self.current(state)
        self.memory_mgr.update(state.decision.tool_name, state.decision.tool_args, result)
        business_state = snapshot_state(self.db_path)
        self.oscillation_guard.observe(state.decision.tool_name, state.decision.tool_args, result,
            self.memory_mgr.get_snapshot(), business_state)
        if not result.ok:
            task.errors.append({'step': state.steps, 'code': result.error_code, 'error': result.error})
        if result.ok and state.decision.tool_name == 'execute_python':
            data = result.data or {}
            code_hash = hashlib.sha256(state.decision.tool_args['code'].encode()).hexdigest()
            if (data.get('ok') is True and data.get('stage') == 'full' and data.get('run_id') == state.run_id
                and data.get('code_sha256') == code_hash and isinstance(data.get('metrics'), dict)
                and isinstance(data.get('summary'), str) and isinstance(data.get('tables'), dict)):
                self.python_results[task.task_id] = deepcopy(data)
                task.artifacts = deepcopy(data.get('artifacts', []))
                if task.verification_capability == 'python':
                    canonical = {key: deepcopy(data[key]) for key in ('summary', 'metrics', 'tables')}
                    task.authoritative_python_result = {'task_id': task.task_id, 'run_id': state.run_id,
                        'tool': 'execute_python', 'capability': 'python', 'stage': 'full', 'code_sha256': code_hash,
                        'input_hashes': deepcopy(data.get('input_hashes', {})), 'inputs': deepcopy(data.get('inputs', [])),
                        'canonical_result': canonical, 'artifacts': deepcopy(task.artifacts)}
                    task.result = deepcopy(canonical)
        self._emit_event(state.run_id, 'MEMORY_UPDATE', self.memory_mgr.get_snapshot())
        self.publish_tasks(state, state.tasks)
        data = result.data if isinstance(result.data, dict) else {}
        commit_state = (result.evidence or {}).get('commit_state', data.get('commit_state'))
        confirmed = (result.ok and not result.retriable and commit_state not in ('unknown', 'not_committed')
                     and (data.get('outcome') == 'confirmed_mutation' or commit_state == 'committed'))
        previous = self.mutation_checkpoint_states.get(task.task_id, self.task_pre_states[task.task_id])
        # A repeated commit receipt with unchanged persisted state cannot rearm a checkpoint.
        if confirmed and not no_mutations(previous, business_state):
            self.mutation_checkpoint_states[task.task_id] = deepcopy(business_state)
            self._emit_event(state.run_id, 'VERIFICATION_CHECKPOINT', {'step': state.steps, 'task_id': task.task_id,
                'tool': state.decision.tool_name, 'outcome': data.get('outcome'), 'commit_state': commit_state,
                'evidence': deepcopy(result.evidence)})
            return {'tasks': state.tasks, 'route': 'verify'}
        return {'tasks': state.tasks, 'route': 'context_build'}

    async def verify(self, state):
        task = self.current(state)
        task.verification_attempts += 1
        objective = state.objective if len(state.tasks) == 1 else task.goal
        self._update_run_status(state.run_id, 'verifying')
        self._emit_event(state.run_id, 'VERIFICATION_REQUESTED', {'task_id': task.task_id, 'attempt': task.verification_attempts})
        result = await self.capability_verifier.verify(objective, task, self.task_pre_states[task.task_id], self.memory_mgr, self.python_results.get(task.task_id))
        self._emit_event(state.run_id, 'VERIFICATION', {'task_id': task.task_id, **result.model_dump()})
        if result.verified:
            task.status = TaskStatus.COMPLETED
            task.result['verification'] = result.model_dump()
            self.publish_tasks(state, state.tasks)
            return {'tasks': state.tasks, 'verification': result, 'current_task_id': None, 'route': 'select_subtask'}
        if result.outcome == 'AMBIGUITY':
            task.status = TaskStatus.NEEDS_CLARIFICATION
            return {'tasks': state.tasks, 'verification': result, 'route': 'finish', 'terminal': {'run_id': state.run_id, 'status': 'waiting_for_clarification', 'question': result.summary}}
        if result.outcome == 'FATAL_FAILURE' or task.verification_attempts >= self.max_verification_attempts:
            task.status = TaskStatus.FAILED
            return {'tasks': state.tasks, 'verification': result, 'route': 'finish', 'terminal': {'run_id': state.run_id, 'status': 'failed', 'error': result.summary}}
        return {'tasks': state.tasks, 'verification': result, 'route': 'context_build', 'observation': {'verification_failed': True, 'discrepancies': result.discrepancies}}

    async def finish(self, state):
        terminal = state.terminal
        if terminal is None:
            if not state.tasks or any(task.status != TaskStatus.COMPLETED or not task.result.get('verification', {}).get('verified') for task in state.tasks):
                terminal = {'run_id': state.run_id, 'status': 'failed', 'error': 'Unverified task graph cannot complete'}
            else:
                after = snapshot_state(self.db_path)
                if len(state.tasks) == 1:
                    coverage = self.capability_verifier.audit_mutations(state.tasks, self.pre_state, after)
                    if coverage.verified:
                        coverage = Verification.model_validate(state.tasks[0].result['verification'])
                else:
                    coverage = await self.capability_verifier.verify_coverage(state.objective, state.tasks, self.pre_state, after)
                state.verification = coverage
                self._emit_event(state.run_id, 'VERIFICATION', {'stage': 'original_objective', **coverage.model_dump()})
                terminal = {'run_id': state.run_id, 'status': 'completed' if coverage.verified else 'failed', 'steps': state.steps}
                if not coverage.verified:
                    terminal['error'] = coverage.summary
        verification = state.verification
        payload = verification.evidence.get('verification_result', verification.model_dump()) if verification else None
        if terminal['status'] != 'completed' and payload and payload.get('verified'):
            payload = {'verified': False, 'summary': 'The original objective remains incomplete.', 'evidence': {'last_verified_subtask': payload}}
        if verification and verification.verified and terminal['status'] == 'completed':
            terminal['verification'] = payload
        self._update_run_status(state.run_id, terminal['status'], verification=payload, error=terminal.get('error') or terminal.get('question'))
        self._emit_event(state.run_id, 'COMPLETE' if terminal['status'] == 'completed' else 'CLARIFICATION_NEEDED' if terminal['status'] == 'waiting_for_clarification' else 'FAILURE', {'summary': terminal.get('error') or terminal.get('question') or verification.summary, **terminal})
        self.publish_final_tasks(state, terminal)
        return {'terminal': terminal}

    def publish_final_tasks(self, state, terminal):
        for task in state.tasks:
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.FAILED
                task.errors.append({'step': self.step_count, 'code': 'RUN_FAILED', 'error': terminal.get('error') or terminal.get('question') or 'Execution interrupted'})
            elif task.status == TaskStatus.PENDING:
                task.status = TaskStatus.BLOCKED
        self.last_tasks = state.tasks
        self._emit_event(state.run_id, 'TASK_GRAPH', {'tasks': [task.model_dump() for task in state.tasks]})

    def _finish_report(self, objective, run_id, result):
        for task in getattr(self, 'last_tasks', []):
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.FAILED
                task.errors.append({'step': self.step_count, 'code': 'RUN_FAILED', 'error': result.get('error') or result.get('question') or 'Execution interrupted'})
            elif task.status == TaskStatus.PENDING:
                task.status = TaskStatus.BLOCKED
        self._emit_event(run_id, 'TASK_GRAPH', {'tasks': [task.model_dump() for task in getattr(self, 'last_tasks', [])]})
        return super()._finish_report(objective, run_id, result)

    def _report_fields(self):
        tasks = [task.model_dump() for task in getattr(self, 'last_tasks', [])]
        return {'runtime': 'v2', 'tasks': tasks, 'artifacts': [artifact for task in tasks for artifact in task['artifacts']],
                'usage': self.provider.usage_snapshot()}
