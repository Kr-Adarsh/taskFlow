"""Validated task graph and shared LangGraph state."""
from enum import Enum
from typing import Any, Literal
from pydantic import Field, model_validator
from backend.app.agent.schemas import StrictModel, AgentDecision


class TaskStatus(str, Enum):
    PENDING = 'PENDING'
    RUNNING = 'RUNNING'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'
    BLOCKED = 'BLOCKED'
    NEEDS_CLARIFICATION = 'NEEDS_CLARIFICATION'


class SourceReference(StrictModel):
    document_id: str
    chunk_id: str
    quote: str


class PlannedTask(StrictModel):
    task_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,60}$')
    goal: str = Field(min_length=1)
    dependencies: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(min_length=1)
    verification_capability: Literal['browser', 'documents', 'python']


class Subtask(PlannedTask):
    status: TaskStatus = TaskStatus.PENDING
    result: dict[str, Any] = Field(default_factory=dict)
    evidence: list[SourceReference] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    verification_attempts: int = 0
    authoritative_python_result: dict[str, Any] | None = None
    model_reported_result: dict[str, Any] = Field(default_factory=dict)
    model_result_matches_authoritative: Literal[True, False, 'unknown'] = 'unknown'


class TaskPlan(StrictModel):
    objective: str
    success_criteria: list[str] = Field(min_length=1)
    tasks: list[PlannedTask] = Field(min_length=1, max_length=12)

    @model_validator(mode='after')
    def valid_dependencies(self):
        ids = {task.task_id for task in self.tasks}
        if len(ids) != len(self.tasks):
            raise ValueError('Task IDs must be unique')
        visiting, visited = set(), set()
        tasks = {task.task_id: task for task in self.tasks}
        def visit(task_id):
            if task_id in visiting:
                raise ValueError('Task dependencies contain a cycle')
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in tasks[task_id].dependencies:
                if dependency not in ids:
                    raise ValueError('Unknown task dependency')
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)
        for task_id in ids:
            visit(task_id)
        return self


class TaskSelection(StrictModel):
    task_id: str
    explanation: str


class Decision(AgentDecision):
    evidence: list[SourceReference] = Field(default_factory=list)
    result: dict[str, Any] = Field(default_factory=dict)
    replan: bool = False

    @model_validator(mode='after')
    def replan_without_action(self):
        if self.replan and self.action.value != 'ready_for_verification':
            raise ValueError('Replan uses ready_for_verification without dispatching a tool')
        return self


class Verification(StrictModel):
    outcome: Literal['PASS', 'RECOVERABLE_FAILURE', 'AMBIGUITY', 'FATAL_FAILURE']
    verified: bool
    summary: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    discrepancies: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def consistent(self):
        if self.verified != (self.outcome == 'PASS'):
            raise ValueError('Only PASS is verified')
        return self


class GraphState(StrictModel):
    run_id: str
    objective: str
    success_criteria: list[str] = Field(default_factory=list)
    tasks: list[Subtask] = Field(default_factory=list)
    current_task_id: str | None = None
    observation: dict[str, Any] = Field(default_factory=dict)
    decision: Decision | None = None
    tool_result: dict[str, Any] | None = None
    capability: str | None = None
    steps: int = 0
    replans: int = 0
    route: str = 'select_subtask'
    terminal: dict[str, Any] | None = None
    verification: Verification | None = None


class AnswerAssessment(StrictModel):
    covered: bool
    reason: str
    material_claims: list[str]
    unsupported_claims: list[str]


class CoverageAssessment(StrictModel):
    covered: bool
    missing_requirements: list[str]
    reason: str
