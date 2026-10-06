"""
Run control API and SSE event streaming for Operon Agent.
Allows starting runs, querying run status, streaming live events, and viewing screenshots.
"""

import asyncio
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Optional, Callable
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field, ConfigDict, field_validator

from backend.app.agent_v2.graph import AgentRunner
from backend.app.agent.provider import get_default_provider
from backend.app.tools.browser_tools import browser_manager
from backend.app.workspace.db import get_db_connection
from backend.app.workspace.lease import reserve_run, WorkspaceBusy

router = APIRouter()

# Active runs in-memory listener queues for live SSE streaming
active_subscribers: dict[str, list[asyncio.Queue]] = {}

def broadcast_event(run_id: str, event_type: str, payload: dict) -> None:
    queues = active_subscribers.get(run_id, [])
    event_data = {
        "run_id": run_id,
        "event_type": event_type,
        "payload": payload,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    for q in queues:
        q.put_nowait(event_data)

class StartRunPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    objective: str = Field(min_length=1, max_length=4000)
    run_id: Optional[str] = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,80}$")
    max_steps: int = Field(default=20, ge=1, le=40)

    @field_validator("objective")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Objective cannot be empty")
        return value

_runner_factory: Optional[Callable[..., AgentRunner]] = None

def set_runner_factory(factory: Optional[Callable[..., AgentRunner]]) -> None:
    global _runner_factory
    _runner_factory = factory

def get_runner(max_steps: int = 20) -> AgentRunner:
    if _runner_factory:
        return _runner_factory(max_steps=max_steps)
    return AgentRunner(
        provider=get_default_provider(),
        max_steps=max_steps,
        event_callback=broadcast_event
    )

@router.post("/api/runs")
async def api_start_run(payload: StartRunPayload, background_tasks: BackgroundTasks):
    import uuid
    run_id = payload.run_id or f"run_{uuid.uuid4().hex[:10]}"

    runner = get_runner(max_steps=payload.max_steps)

    try:
        reserve_run(payload.objective, run_id, runner.db_path)
    except WorkspaceBusy as error:
        raise HTTPException(status_code=409, detail=str(error))
    background_tasks.add_task(runner.execute_task, payload.objective, run_id, reserved=True)

    return {"ok": True, "run_id": run_id, "status": "planning", "objective": payload.objective}

@router.get("/api/runs/{run_id}")
def api_get_run(run_id: str):
    with get_db_connection() as conn:
        run_row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if not run_row:
            raise HTTPException(status_code=404, detail="Run not found")
        
        events_rows = conn.execute(
            "SELECT * FROM run_events WHERE run_id = ? ORDER BY id ASC",
            (run_id,)
        ).fetchall()

    events = [
        {
            "id": r["id"],
            "event_type": r["event_type"],
            "payload": json.loads(r["payload"]),
            "timestamp": r["timestamp"]
        }
        for r in events_rows
    ]

    return {
        "run_id": run_row["run_id"],
        "objective": run_row["objective"],
        "status": run_row["status"],
        "plan": json.loads(run_row["plan"]) if run_row["plan"] else None,
        "working_memory": json.loads(run_row["working_memory"]) if run_row["working_memory"] else None,
        "verification_result": json.loads(run_row["verification_result"]) if run_row["verification_result"] else None,
        "error": run_row["error"],
        "created_at": run_row["created_at"],
        "updated_at": run_row["updated_at"],
        "report": next((event["payload"] for event in reversed(events) if event["event_type"] == "FINAL_REPORT"), None),
        "events": events
    }

TERMINAL_STATES = {"completed", "failed", "waiting_for_clarification", "interrupted"}

async def stream_events(run_id, request, cursor=0):
    while not await request.is_disconnected():
        with get_db_connection(read_only=True) as connection:
            run = connection.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()
            events = connection.execute("SELECT * FROM run_events WHERE run_id=? AND id>? ORDER BY id", (run_id,cursor)).fetchall()
            has_report = connection.execute("SELECT 1 FROM run_events WHERE run_id=? AND event_type='FINAL_REPORT'", (run_id,)).fetchone()
            owns_workspace = connection.execute('SELECT 1 FROM workspace_lease WHERE run_id=?', (run_id,)).fetchone()
        for row in events:
            cursor = row["id"]
            event = {"id": cursor, "run_id": run_id, "event_type": row["event_type"], "payload": json.loads(row["payload"]), "timestamp": row["timestamp"]}
            yield f"id: {cursor}\ndata: {json.dumps(event)}\n\n"
        if not run or (run["status"] in TERMINAL_STATES and (has_report or not owns_workspace)):
            yield "data: " + json.dumps({"run_id": run_id, "event_type": "STREAM_END", "payload": {"status": run["status"] if run else "interrupted"}, "timestamp": datetime.now(timezone.utc).isoformat()}) + "\n\n"
            return
        yield ": heartbeat\n\n"
        await asyncio.sleep(0.25)

@router.get("/api/runs/{run_id}/stream")
async def api_stream_run_events(run_id: str, request: Request, cursor: int = 0):
    with get_db_connection(read_only=True) as connection:
        if not connection.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone():
            raise HTTPException(status_code=404, detail="Run not found")
    try:
        cursor = max(0, cursor, int(request.headers.get("last-event-id", "0")))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid event cursor")
    return StreamingResponse(stream_events(run_id, request, cursor), media_type="text/event-stream", headers={"Cache-Control":"no-cache", "X-Accel-Buffering":"no"})

@router.get("/api/runs/{run_id}/screenshot")
def api_get_latest_screenshot(run_id: str):
    with get_db_connection(read_only=True) as connection:
        if not connection.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone():
            raise HTTPException(status_code=404, detail="Run not found")
        events = connection.execute("SELECT payload FROM run_events WHERE run_id=? AND event_type='OBSERVATION' ORDER BY id DESC", (run_id,)).fetchall()
    path = None
    for event in events:
        evidence = json.loads(event["payload"]).get("evidence") or {}
        if evidence.get("screenshot"):
            path = evidence["screenshot"]
            break
    if not path or not Path(path).exists() or Path(path).parent != browser_manager.screenshot_root / run_id:
        raise HTTPException(status_code=404, detail="No screenshot for this run")
    return FileResponse(path, media_type="image/png")


@router.get('/api/runs/{run_id}/artifacts/{analysis_id}/{filename}')
def api_get_artifact(run_id: str, analysis_id: str, filename: str):
    import hashlib
    from backend.app.capabilities.python.sandbox import artifact_root
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run_id) or not re.fullmatch(r'analysis_[a-f0-9]{12}', analysis_id) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', filename):
        raise HTTPException(status_code=404, detail='Artifact not found')
    reference = f'/api/runs/{run_id}/artifacts/{analysis_id}/{filename}'
    with get_db_connection(read_only=True) as connection:
        events = connection.execute("SELECT payload FROM run_events WHERE run_id=? AND event_type='OBSERVATION'", (run_id,)).fetchall()
    artifacts = [artifact for event in events for artifact in (json.loads(event['payload']).get('data') or {}).get('artifacts', [])]
    artifact = next((item for item in artifacts if item['reference'] == reference), None)
    path = artifact_root() / run_id / analysis_id / filename
    if not artifact or path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(artifact_root() / run_id):
        raise HTTPException(status_code=404, detail='Artifact not found')
    if hashlib.sha256(path.read_bytes()).hexdigest() != artifact['sha256']:
        raise HTTPException(status_code=409, detail='Artifact changed after execution')
    return FileResponse(path, filename=filename)
