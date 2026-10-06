"""
Tests for Runs API and event persistence.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.workspace.seed import reset_demo_env
from backend.app.api.runs import set_runner_factory, broadcast_event
from backend.app.agent.loop import AgentRunner
from backend.app.agent.provider import FakeProvider
from backend.app.agent.schemas import TaskPlan, AgentDecision, AgentActionType, VerificationIntent
from backend.app.agent.verifier import VerifierEngine

@pytest.fixture(autouse=True)
def setup_test():
    reset_demo_env()
    fake = FakeProvider([
        TaskPlan(objective="Test", success_criteria=["Criteria 1"], strategy=["Strategy 1"]),
        AgentDecision(thought="Done", action=AgentActionType.READY_FOR_VERIFICATION)
    ])
    set_runner_factory(lambda max_steps=20: AgentRunner(provider=fake, verifier=VerifierEngine(intent=VerificationIntent(collection="unsupported")), max_steps=max_steps, event_callback=broadcast_event))
    yield
    set_runner_factory(None)

client = TestClient(app)

def test_run_creation_and_retrieval():
    payload = {
        "objective": "Find the latest invoice from Acme Corp and enter into Finance",
        "run_id": "test_run_123"
    }
    res = client.post("/api/runs", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["run_id"] == "test_run_123"

    # Query details
    get_res = client.get("/api/runs/test_run_123")
    assert get_res.status_code == 200
    run_data = get_res.json()
    assert run_data["objective"] == payload["objective"]
    assert "events" in run_data
    assert len(run_data["events"]) >= 2
