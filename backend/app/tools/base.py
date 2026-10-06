"""
Tool contracts and execution envelopes for Operon.
"""

from typing import Any, Callable, Optional
from pydantic import BaseModel, Field, ConfigDict

class ToolResult(BaseModel):
    ok: bool
    data: Any = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    retriable: bool = False
    evidence: Optional[dict] = None

class ToolDefinition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    parameters: dict = Field(default_factory=dict)
    func: Optional[Callable] = None
