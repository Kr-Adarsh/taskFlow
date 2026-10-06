"""
Pydantic contracts and schemas for TaskFlow Agent runtime.
Defines AgentDecision, TaskPlan, WorkingMemory, and VerificationResult.
"""

from enum import Enum
from typing import Any, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator

class AgentActionType(str, Enum):
    ACT = "act"
    READY_FOR_VERIFICATION = "ready_for_verification"
    NEED_CLARIFICATION = "need_clarification"
    FAIL = "fail"

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentDecision(StrictModel):
    """
    Structured decision made by the autonomous agent on each execution step.
    Must adhere strictly to this schema.
    """
    thought: str = Field(..., description="Short explanation of why this action was chosen based on the observation")
    action: AgentActionType = Field(..., description="Decision type: act, ready_for_verification, need_clarification, fail")
    tool_name: Optional[str] = Field(default=None, description="Name of tool to execute if action is 'act'")
    tool_args: Optional[dict[str, Any]] = Field(default=None, description="Arguments for the selected tool")
    clarification_question: Optional[str] = Field(default=None, description="Question for the user if clarification is required")
    failure_reason: Optional[str] = Field(default=None, description="Reason if action is 'fail'")

    @model_validator(mode="after")
    def validate_action(self):
        if self.action == AgentActionType.ACT:
            if not self.tool_name or self.tool_args is None:
                raise ValueError("act requires tool_name and tool_args")
            if self.clarification_question is not None or self.failure_reason is not None:
                raise ValueError("act cannot include clarification or failure fields")
        elif self.tool_name is not None or self.tool_args is not None:
            raise ValueError("non-act decisions cannot include tools")
        if self.action == AgentActionType.NEED_CLARIFICATION and not self.clarification_question:
            raise ValueError("need_clarification requires a question")
        if self.action == AgentActionType.FAIL and not self.failure_reason:
            raise ValueError("fail requires a reason")
        if self.action != AgentActionType.NEED_CLARIFICATION and self.clarification_question is not None:
            raise ValueError("clarification_question is only valid for need_clarification")
        if self.action != AgentActionType.FAIL and self.failure_reason is not None:
            raise ValueError("failure_reason is only valid for fail")
        return self


class TaskPlan(StrictModel):
    """
    Initial task plan formulated from the user's natural language objective.
    """
    objective: str = Field(..., description="Normalized objective")
    success_criteria: list[str] = Field(..., min_length=1, description="List of concrete, observable success criteria")
    strategy: list[str] = Field(..., description="High-level sequence of expected milestones")

class WorkingMemory(StrictModel):
    """
    Structured working memory maintained and updated during the execution loop.
    """
    sources: dict[str, Any] = Field(default_factory=dict)
    pages: dict[str, Any] = Field(default_factory=dict)
    selected_source: Optional[str] = None
    extracted_facts: dict[str, Any] = Field(default_factory=dict, description="Key facts extracted from documents or pages")
    source_references: dict[str, str] = Field(default_factory=dict, description="Mapping of facts to source documents/pages")
    recent_notes: list[str] = Field(default_factory=list, description="Key observations or transient errors encountered")

class CriterionResult(StrictModel):
    criterion: str
    passed: bool
    evidence: dict[str, Any] = Field(default_factory=dict)
    discrepancy: Optional[str] = None

class VerificationResult(StrictModel):
    """
    Independent verification result evaluated against read-only ground truth.
    """
    verified: bool
    summary: str
    criteria_results: list[CriterionResult] = Field(default_factory=list)
    discrepancies: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class VerificationIntent(StrictModel):
    collection: str = Field(pattern=r"^(invoices|tickets|accounts|unsupported)$")
    company: Optional[str] = Field(default=None, description="Exact company named in the original objective, including all suffixes/numbers; null if no company is named. Never abbreviate identity.")
    selection: Literal["latest", "specific", "unresolved"] = Field(default="specific", description="latest for a latest-source request; specific for an explicitly identified source; unresolved for another unsupported rule")
    invoice_number: Optional[str] = None
    complaint_id: Optional[str] = Field(default=None, description="Exact source identifier named in the original request for a source-linked ticket, including a complaint number when provided. Reading the source and extracting its customer are supported evidence checks. Required for tickets even when the source is mentioned in a procedural clause.")
    priority: Optional[str] = Field(default=None, pattern=r"^(High|Medium|Low)$")
    condition_tier: Optional[str] = Field(default=None, description="Exact CRM tier required by a conditional ticket-creation request; null for unconditional creation. CRM tier lookup is supported.")
    require_new_record: bool = True
    requested_fields: list[Literal["company", "invoice_number", "currency", "amount_minor", "due_date", "source_reference", "customer", "priority", "summary", "customer_name", "tier", "mrr", "account_manager", "status"]] = Field(default_factory=list)
    unsupported_criteria: list[str] = Field(default_factory=list, description="Original-request requirements outside the verifier's declared capabilities; not missing execution evidence or extra planner suggestions.")


class SummaryAssessment(StrictModel):
    accurate: bool
    reason: str
    source_quotes: list[str]
    contradictions: list[str]
