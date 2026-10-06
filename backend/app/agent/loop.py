"""
Closed-loop autonomous agent runtime for Operon.
Implements the plan -> observe -> decide -> act -> observe -> memory -> verify cycle.
Handles bounded step budgets, event emission, SSE streaming, and verification repair.
"""

from datetime import datetime, timezone
import asyncio
import os
import json
from pathlib import Path
import time
from typing import Any, Callable, Optional
import uuid

from backend.app.agent.schemas import (
    AgentActionType,
    AgentDecision,
    TaskPlan,
    WorkingMemory,
    VerificationResult,
)
from backend.app.agent.provider import LLMProvider, get_default_provider, ProviderError
from backend.app.agent.prompts import build_planner_prompt, build_executor_prompt
from backend.app.agent.memory import MemoryManager
from backend.app.agent.verifier import VerifierEngine, snapshot_state, read_sources
from backend.app.tools.registry import ToolRegistry, build_default_tool_registry
from backend.app.tools.base import ToolResult
from backend.app.workspace.db import get_db_connection
from backend.app.workspace.lease import reserve_run, release_run, mutation_owner
from backend.app.tools.browser_tools import browser_manager

class AgentRunner:
    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        tool_registry: Optional[ToolRegistry] = None,
        verifier: Optional[VerifierEngine] = None,
        max_steps: int = 20,
        max_verification_attempts: int = 2,
        db_path: Optional[Path] = None,
        event_callback: Optional[Callable[[str, str, dict], None]] = None,
        run_deadline_seconds: float | None = None
    ):
        if not 1 <= max_steps <= 40 or not 1 <= max_verification_attempts <= 3:
            raise ValueError("Invalid runtime bounds")
        self.run_deadline_seconds = float(run_deadline_seconds if run_deadline_seconds is not None else os.getenv("OPERON_RUN_DEADLINE_SECONDS", "600"))
        if not 0 < self.run_deadline_seconds <= 600:
            raise ValueError("Invalid run deadline")
        self.provider = provider or get_default_provider()
        self.registry = tool_registry or build_default_tool_registry()
        self.provider.configure_tools(self.registry.get_schemas())
        self.verifier = verifier or VerifierEngine(db_path=db_path, provider=self.provider)
        self.max_steps = max_steps
        self.max_verification_attempts = max_verification_attempts
        self.db_path = db_path
        self.event_callback = event_callback

    def _emit_event(self, run_id: str, event_type: str, payload: dict) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        
        # Persist event to DB
        with get_db_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO run_events (run_id, event_type, payload, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, event_type, json.dumps(payload), now_iso)
            )
            conn.commit()

        if self.event_callback:
            self.event_callback(run_id, event_type, payload)

    async def execute_task(self, objective: str, run_id: Optional[str] = None, reserved: bool = False) -> dict[str, Any]:
        run_id = run_id or f"run_{uuid.uuid4().hex[:10]}"
        if reserved:
            with get_db_connection(self.db_path) as connection:
                row = connection.execute("SELECT token FROM workspace_lease WHERE run_id=?", (run_id,)).fetchone()
            if not row:
                raise ValueError("Run does not own the workspace")
            token = row["token"]
        else:
            token = reserve_run(objective, run_id, self.db_path)
        self.started = time.perf_counter()
        self.step_count = 0
        self.memory_mgr = MemoryManager()
        context_token = mutation_owner.set((run_id, token))
        try:
            await browser_manager.bind_run(run_id, token)
            async with asyncio.timeout(self.run_deadline_seconds):
                result = await self._execute_owned(objective, run_id)
            return self._finish_report(objective, run_id, result)
        except asyncio.CancelledError:
            self._update_run_status(run_id, "interrupted", error="Run cancelled before completion")
            self._emit_event(run_id, "FAILURE", {"error": "Run cancelled before completion"})
            self._finish_report(objective, run_id, {"run_id": run_id, "status": "interrupted", "error": "Run cancelled before completion"})
            raise
        except Exception as error:
            message = "Run deadline exceeded" if isinstance(error, TimeoutError) else str(error) if isinstance(error, ProviderError) else f"Run infrastructure failure: {type(error).__name__}"
            self._update_run_status(run_id, "failed", error=message)
            self._emit_event(run_id, "FAILURE", {"error": message})
            return self._finish_report(objective, run_id, {"run_id": run_id, "status": "failed", "error": message})
        finally:
            try:
                try:
                    await asyncio.wait_for(browser_manager.close(), timeout=5)
                except Exception:
                    self._emit_event(run_id, "CLEANUP_ERROR", {"error": "Browser cleanup failed"})
            finally:
                browser_manager.run_id = browser_manager._lease_token = None
                release_run(run_id, self.db_path)
                mutation_owner.reset(context_token)

    async def _execute_owned(self, objective: str, run_id: str) -> dict[str, Any]:
        self._emit_event(run_id, "TASK", {"objective": objective})

        pre_state = snapshot_state(self.db_path)
        self.pre_state = pre_state

        # --- Step 1: Formulate TaskPlan ---
        try:
            planner_msgs = build_planner_prompt(objective)
            plan, plan_meta = await self.provider.generate_structured(planner_msgs, TaskPlan)
        except Exception as e:
            err_msg = f"Planning failed: {str(e) if isinstance(e, ProviderError) else type(e).__name__}"
            self._emit_event(run_id, "FAILURE", {"error": err_msg})
            self._update_run_status(run_id, "failed", error=err_msg)
            return {"run_id": run_id, "status": "failed", "error": err_msg}

        self._emit_event(run_id, "MODEL_CALL", {"stage": "planner", "metadata": plan_meta})
        self._emit_event(run_id, "PLAN", plan.model_dump())
        self._update_run_status(run_id, "running", plan=plan.model_dump())

        # --- Step 2: Execution Loop ---
        memory_mgr = self.memory_mgr
        recent_actions: list[dict[str, Any]] = []
        current_observation: dict[str, Any] = {
            "status": "Ready",
            "message": "Task started. Workspace apps are available at /workspace/finance, /workspace/crm, /workspace/support, /workspace/documents."
        }
        
        verification_attempts = 0
        step_count = 0
        final_verification: Optional[VerificationResult] = None

        while step_count < self.max_steps:
            step_count += 1
            self.step_count = step_count
            tool_schemas = self.registry.get_schemas()

            prompt_msgs = build_executor_prompt(
                objective=objective,
                success_criteria=plan.success_criteria,
                working_memory=memory_mgr.get_snapshot(),
                recent_actions=recent_actions,
                current_observation=current_observation,
                tool_schemas=tool_schemas
            )

            try:
                decision, meta = await self.provider.generate_structured(prompt_msgs, AgentDecision)
            except Exception as error:
                message = f"Decision generation failed at step {step_count}: {str(error) if isinstance(error, ProviderError) else type(error).__name__}"
                self._emit_event(run_id, "FAILURE", {"error": message, "step": step_count})
                self._update_run_status(run_id, "failed", error=message)
                return {"run_id": run_id, "status": "failed", "error": message}

            # Handle Action Types
            if decision.action == AgentActionType.ACT:
                tool_name = decision.tool_name or ""
                tool_args = decision.tool_args or {}

                self._emit_event(run_id, "ACTION", {
                    "step": step_count,
                    "thought": decision.thought,
                    "tool": tool_name,
                    "args": tool_args,
                    "latency": meta.get("latency_seconds"),
                    "model_metadata": meta
                })

                # Execute tool
                async with asyncio.timeout(20):
                    tool_result = await self.registry.execute(tool_name, tool_args)

                # Update memory
                memory_mgr.update_from_tool_result(tool_name, tool_args, tool_result)
                self._emit_event(run_id, "OBSERVATION", {
                    "step": step_count,
                    "ok": tool_result.ok,
                    "data": tool_result.data,
                    "error": tool_result.error,
                    "retriable": tool_result.retriable,
                    "error_code": tool_result.error_code,
                    "evidence": tool_result.evidence
                })
                self._emit_event(run_id, "MEMORY_UPDATE", memory_mgr.get_snapshot())

                # Record in history
                recent_actions.append({
                    "step": step_count,
                    "thought": decision.thought,
                    "tool": tool_name,
                    "args": tool_args,
                    "ok": tool_result.ok,
                    "error": tool_result.error
                })

                if tool_result.ok:
                    current_observation = tool_result.data or {}
                else:
                    current_observation = {
                        "error": tool_result.error,
                        "error_code": tool_result.error_code,
                        "retriable": tool_result.retriable,
                        "data": tool_result.data,
                        "evidence": tool_result.evidence
                    }

            elif decision.action == AgentActionType.READY_FOR_VERIFICATION:
                verification_attempts += 1
                self._emit_event(run_id, "VERIFICATION_REQUESTED", {
                    "thought": decision.thought,
                    "attempt": verification_attempts
                })
                self._update_run_status(run_id, "verifying")

                # Run independent verification
                final_verification = await self.verifier.verify_run(objective, memory_mgr.get_snapshot(), success_criteria=plan.success_criteria, pre_state=pre_state, post_state=snapshot_state(self.db_path), source_evidence=read_sources(self.db_path))
                
                self._update_run_status(run_id, "verifying", verification=final_verification.model_dump())
                self._emit_event(run_id, "VERIFICATION", final_verification.model_dump())

                if final_verification.verified:
                    self._emit_event(run_id, "COMPLETE", {
                        "summary": final_verification.summary,
                        "steps": step_count,
                        "verification": final_verification.model_dump()
                    })
                    self._update_run_status(
                        run_id,
                        "completed",
                        working_memory=memory_mgr.get_snapshot(),
                        verification=final_verification.model_dump()
                    )
                    return {
                        "run_id": run_id,
                        "status": "completed",
                        "steps": step_count,
                        "verification": final_verification.model_dump()
                    }
                else:
                    # Verification failed: feed back discrepancies to loop for repair if budget permits
                    if verification_attempts < self.max_verification_attempts:
                        self._emit_event(run_id, "RETRY", {
                            "reason": "Verification discrepancies returned to executor for bounded repair",
                            "discrepancies": final_verification.discrepancies
                        })
                        current_observation = {
                            "verification_failed": True,
                            "discrepancies": final_verification.discrepancies,
                            "verification_evidence": final_verification.model_dump()
                        }
                        self._update_run_status(run_id, "running")
                    else:
                        fail_msg = f"Verification exhausted ({verification_attempts} attempts): {', '.join(final_verification.discrepancies)}"
                        self._emit_event(run_id, "FAILURE", {"error": fail_msg})
                        self._update_run_status(
                            run_id,
                            "failed",
                            error=fail_msg,
                            verification=final_verification.model_dump()
                        )
                        return {"run_id": run_id, "status": "failed", "error": fail_msg}

            elif decision.action == AgentActionType.NEED_CLARIFICATION:
                q = decision.clarification_question or "Clarification needed to proceed."
                self._emit_event(run_id, "CLARIFICATION_NEEDED", {"question": q})
                self._update_run_status(run_id, "waiting_for_clarification", error=q)
                return {"run_id": run_id, "status": "waiting_for_clarification", "question": q}

            elif decision.action == AgentActionType.FAIL:
                reason = decision.failure_reason or "Agent determined task cannot be completed."
                self._emit_event(run_id, "FAILURE", {"error": reason})
                self._update_run_status(run_id, "failed", error=reason)
                return {"run_id": run_id, "status": "failed", "error": reason}

        # Step budget exhausted
        exhausted_msg = f"Execution exceeded step budget ({self.max_steps} steps)"
        self._emit_event(run_id, "FAILURE", {"error": exhausted_msg})
        self._update_run_status(run_id, "failed", error=exhausted_msg)
        return {"run_id": run_id, "status": "failed", "error": exhausted_msg}

    def _finish_report(self, objective, run_id, result):
        from backend.app.agent.verifier import state_delta
        with get_db_connection(self.db_path, read_only=True) as connection:
            row = connection.execute("SELECT verification_result FROM runs WHERE run_id=?", (run_id,)).fetchone()
        verification = json.loads(row["verification_result"]) if row and row["verification_result"] else None
        report = {"objective": objective, "status": result["status"], "steps": self.step_count,
                  "duration_seconds": round(time.perf_counter()-self.started, 3),
                  "provider": type(self.provider).__name__, "model": getattr(self.provider, "model", "fake"),
                  "verification": verification, "error": result.get("error"), "question": result.get("question"),
                  "state_delta": state_delta(self.pre_state, snapshot_state(self.db_path)) if hasattr(self, "pre_state") else None}
        report.update(self._report_fields())
        self._emit_event(run_id, "FINAL_REPORT", report)
        return {**result, "steps": self.step_count, "report": report}

    def _report_fields(self):
        return {}

    def _update_run_status(
        self,
        run_id: str,
        status: str,
        plan: Optional[dict] = None,
        working_memory: Optional[dict] = None,
        verification: Optional[dict] = None,
        error: Optional[str] = None
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        if working_memory is None and hasattr(self, "memory_mgr"):
            working_memory = self.memory_mgr.get_snapshot()
        with get_db_connection(self.db_path) as conn:
            updates = ["status = ?", "updated_at = ?"]
            params: list[Any] = [status, now_iso]

            if plan is not None:
                updates.append("plan = ?")
                params.append(json.dumps(plan))
            if working_memory is not None:
                updates.append("working_memory = ?")
                params.append(json.dumps(working_memory))
            if verification is not None:
                updates.append("verification_result = ?")
                params.append(json.dumps(verification))
            if error is not None:
                updates.append("error = ?")
                params.append(error)

            params.append(run_id)
            conn.execute(f"UPDATE runs SET {', '.join(updates)} WHERE run_id = ?", params)
            conn.commit()
