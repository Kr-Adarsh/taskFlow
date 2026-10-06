"""
Deterministic unit tests for AgentRunner closed-loop execution,
including planning, observation feedback, verification rejection and bounded repair.
"""

from pathlib import Path
import pytest

from backend.app.agent.schemas import AgentDecision, AgentActionType, TaskPlan
from backend.app.agent.provider import FakeProvider
from backend.app.agent.loop import AgentRunner
from backend.app.tools.registry import ToolRegistry
from backend.app.tools.base import ToolResult
from backend.app.agent.verifier import VerifierEngine
from backend.app.workspace.seed import seed_workspace, reset_demo_env
from backend.app.workspace.db import get_db_connection

@pytest.fixture
def temp_workspace(tmp_path: Path):
    db_file = tmp_path / "test_loop.db"
    seed_workspace(db_path=db_file, force_reseed=True)
    yield db_file

@pytest.mark.anyio
async def test_agent_loop_success_with_verification(temp_workspace: Path):
    fake_provider = FakeProvider()

    # 1. Plan response
    fake_provider.queue_response(TaskPlan(
        objective="Enter invoice into Finance",
        success_criteria=["Invoice INV-1044 recorded in Finance with correct amount and due date"],
        strategy=["Find invoice", "Record in Finance", "Verify"]
    ))

    # 2. Step 1: Search document
    fake_provider.queue_response(AgentDecision(
        thought="Search for Acme invoice",
        action=AgentActionType.ACT,
        tool_name="dummy_tool",
        tool_args={"arg": "search"}
    ))

    # 3. Step 2: Ready for verification
    fake_provider.queue_response(AgentDecision(
        thought="Invoice submitted and visible, requesting verification",
        action=AgentActionType.READY_FOR_VERIFICATION
    ))

    # Tool registry with dummy tool
    registry = ToolRegistry()
    registry.register(
        "dummy_tool",
        "dummy",
        {"type": "object", "properties": {"arg": {"type": "string"}}},
        lambda **kwargs: ToolResult(ok=True, data={"found": "acme_invoice_1044.pdf"})
    )

    # Mock verifier that succeeds
    class MockVerifier(VerifierEngine):
        async def verify_run(self, objective: str, working_memory: dict, **context):
            from backend.app.agent.schemas import VerificationResult, CriterionResult
            return VerificationResult(
                verified=True,
                summary="All criteria verified successfully",
                criteria_results=[CriterionResult(criterion="Invoice recorded", passed=True)]
            )

    runner = AgentRunner(
        provider=fake_provider,
        tool_registry=registry,
        verifier=MockVerifier(db_path=temp_workspace),
        max_steps=5,
        db_path=temp_workspace
    )

    result = await runner.execute_task("Enter Acme invoice")
    assert result["status"] == "completed"
    assert result["verification"]["verified"] is True

@pytest.mark.anyio
async def test_agent_loop_verification_repair_cycle(temp_workspace: Path):
    """
    Tests that a verification failure returns discrepancies to the loop,
    and a second attempt that repairs the issue achieves verified success.
    """
    fake_provider = FakeProvider()

    # Plan
    fake_provider.queue_response(TaskPlan(
        objective="Create invoice",
        success_criteria=["Invoice recorded"],
        strategy=["Submit", "Verify"]
    ))

    # Step 1: Claim ready for verification too early
    fake_provider.queue_response(AgentDecision(
        thought="Attempting verification without doing write",
        action=AgentActionType.READY_FOR_VERIFICATION
    ))

    # Step 2: Model receives discrepancy, executes repair action
    fake_provider.queue_response(AgentDecision(
        thought="Fixing discrepancy by executing repair tool",
        action=AgentActionType.ACT,
        tool_name="repair_tool",
        tool_args={}
    ))

    # Step 3: Requests verification again
    fake_provider.queue_response(AgentDecision(
        thought="Repaired, requesting verification second time",
        action=AgentActionType.READY_FOR_VERIFICATION
    ))

    registry = ToolRegistry()
    registry.register("repair_tool", "repair", {}, lambda: ToolResult(ok=True, data={"repaired": True}))

    class FlakyVerifier(VerifierEngine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = 0

        async def verify_run(self, objective: str, working_memory: dict, **context):
            from backend.app.agent.schemas import VerificationResult, CriterionResult
            self.calls += 1
            if self.calls == 1:
                return VerificationResult(
                    verified=False,
                    summary="Missing invoice record",
                    discrepancies=["Invoice record not found in ledger"]
                )
            return VerificationResult(
                verified=True,
                summary="Repaired invoice verified",
                criteria_results=[CriterionResult(criterion="Invoice recorded", passed=True)]
            )

    runner = AgentRunner(
        provider=fake_provider,
        tool_registry=registry,
        verifier=FlakyVerifier(db_path=temp_workspace),
        max_steps=5,
        max_verification_attempts=2,
        db_path=temp_workspace
    )

    result = await runner.execute_task("Create invoice")
    assert result["status"] == "completed"
    assert result["verification"]["verified"] is True
