"""
Mandatory Early Model Gate: Exercises real Groq (openai/gpt-oss-20b)
against actual TaskPlan and AgentDecision schemas, using real prompts,
registered tool schemas, and multi-step observation/action cycles.
"""

import os
import pytest
import time

from backend.app.agent.schemas import TaskPlan, AgentDecision, AgentActionType
from backend.app.agent.provider import get_default_provider
from backend.app.agent.prompts import build_planner_prompt, build_executor_prompt
from backend.app.tools.registry import build_default_tool_registry
from backend.app.tools.browser_tools import browser_manager
from test_real_acceptance import real_workspace

@pytest.fixture
async def browser_cleanup():
    try:
        yield
    finally:
        await browser_manager.close()

@pytest.mark.skipif(os.getenv("TASKFLOW_RUN_REAL") != "1", reason="Explicit real-model opt-in required")
@pytest.mark.anyio
async def test_model_gate_planner_and_executor(real_workspace, browser_cleanup):
    provider = get_default_provider()
    registry = build_default_tool_registry()
    tool_schemas = registry.get_schemas()

    objective = "Find the latest invoice from Acme Corp, extract the amount and due date, enter it into Finance, and verify that it was recorded correctly."

    # --- Phase 1: Planner Output Schema Test ---
    planner_messages = build_planner_prompt(objective)
    plan_instance, plan_meta = await provider.generate_structured(planner_messages, TaskPlan)

    assert isinstance(plan_instance, TaskPlan)
    assert len(plan_instance.success_criteria) > 0
    assert len(plan_instance.strategy) > 0
    print(f"\n[Model Gate] Planner latency: {plan_meta['latency_seconds']}s")
    print(f"[Model Gate] Plan criteria: {plan_instance.success_criteria}")

    # --- Phase 2: Executor Step 1 (Initial Step) ---
    working_memory = {
        "extracted_facts": {},
        "source_references": {},
        "recent_notes": ["Run started"]
    }
    recent_actions = []
    initial_observation = {
        "status": "Ready",
        "message": "Session started. Available apps: /workspace/finance, /workspace/crm, /workspace/support, /workspace/documents. Document library is available."
    }

    executor_messages_1 = build_executor_prompt(
        objective=objective,
        success_criteria=plan_instance.success_criteria,
        working_memory=working_memory,
        recent_actions=recent_actions,
        current_observation=initial_observation,
        tool_schemas=tool_schemas
    )

    decision_1, meta_1 = await provider.generate_structured(executor_messages_1, AgentDecision)

    assert isinstance(decision_1, AgentDecision)
    assert decision_1.action in [AgentActionType.ACT, AgentActionType.READY_FOR_VERIFICATION]
    assert decision_1.tool_name is not None
    assert registry.get(decision_1.tool_name) is not None, f"Model selected unknown tool: {decision_1.tool_name}"
    
    print(f"[Model Gate] Step 1 latency: {meta_1['latency_seconds']}s")
    print(f"[Model Gate] Step 1 Thought: {decision_1.thought}")
    print(f"[Model Gate] Step 1 Action: {decision_1.tool_name}({decision_1.tool_args})")

    # Dispatch tool
    tool_res_1 = await registry.execute(decision_1.tool_name, decision_1.tool_args or {})
    assert tool_res_1.ok is True, f"Tool execution failed: {tool_res_1.error}"

    # --- Phase 3: Executor Step 2 (Follow-up Observation) ---
    recent_actions.append({
        "step": 1,
        "thought": decision_1.thought,
        "tool": decision_1.tool_name,
        "args": decision_1.tool_args,
        "ok": tool_res_1.ok
    })
    step_2_observation = tool_res_1.data

    executor_messages_2 = build_executor_prompt(
        objective=objective,
        success_criteria=plan_instance.success_criteria,
        working_memory=working_memory,
        recent_actions=recent_actions,
        current_observation=step_2_observation,
        tool_schemas=tool_schemas
    )

    decision_2, meta_2 = await provider.generate_structured(executor_messages_2, AgentDecision)

    assert isinstance(decision_2, AgentDecision)
    assert decision_2.tool_name is not None
    assert registry.get(decision_2.tool_name) is not None
    print(f"[Model Gate] Step 2 latency: {meta_2['latency_seconds']}s")
    print(f"[Model Gate] Step 2 Thought: {decision_2.thought}")
    print(f"[Model Gate] Step 2 Action: {decision_2.tool_name}({decision_2.tool_args})")

    total_latency = plan_meta['latency_seconds'] + meta_1['latency_seconds'] + meta_2['latency_seconds']
    print(f"[Model Gate] Gate Passed! Total 3-call latency: {total_latency:.2f}s")
