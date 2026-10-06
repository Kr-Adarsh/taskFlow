from pathlib import Path
"""All tests, including server fixture setup, use a disposable workspace."""

import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def isolated_workspace(tmp_path_factory):
    previous = os.environ.get("TASKFLOW_DB_PATH")
    screenshot_previous = os.environ.get("TASKFLOW_SCREENSHOTS_DIR")
    os.environ["TASKFLOW_SCREENSHOTS_DIR"] = str(tmp_path_factory.mktemp("screenshots"))
    os.environ["TASKFLOW_DB_PATH"] = str(tmp_path_factory.mktemp("workspace") / "test.db")
    from backend.app.workspace.seed import reset_demo_env
    reset_demo_env()
    yield
    if screenshot_previous is None:
        os.environ.pop("TASKFLOW_SCREENSHOTS_DIR", None)
    else:
        os.environ["TASKFLOW_SCREENSHOTS_DIR"] = screenshot_previous
    if previous is None:
        os.environ.pop("TASKFLOW_DB_PATH", None)
    else:
        os.environ["TASKFLOW_DB_PATH"] = previous


@pytest.fixture
def anyio_backend():
    return "asyncio"



@pytest.fixture(scope="session", autouse=True)
def private_evidence_directories():
    root = Path(__file__).resolve().parents[1] / "docs" / "validation"
    for name in ("current-browser-context-20261006", "verifier-authority-20261006",
                 "python-result-ownership-20261006"):
        (root / name).mkdir(parents=True, exist_ok=True)
