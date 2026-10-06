"""SQLite-backed ownership of the shared browser and workspace."""

from contextvars import ContextVar
from datetime import datetime, timezone
import secrets
import re

from backend.app.workspace.db import get_db_connection

mutation_owner = ContextVar("mutation_owner", default=None)


class WorkspaceBusy(ValueError):
    pass


def reserve_run(objective, run_id, db_path=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ValueError("Invalid run ID")
    if not objective.strip():
        raise ValueError("Objective must not be empty")
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        if connection.execute("SELECT 1 FROM workspace_lease").fetchone():
            raise WorkspaceBusy("Workspace is owned by an active run")
        if connection.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone():
            raise WorkspaceBusy("Run ID already exists")
        connection.execute("INSERT INTO runs(run_id,objective,status,created_at,updated_at) VALUES(?,?,'planning',?,?)", (run_id, objective, now, now))
        connection.execute("INSERT INTO workspace_lease(singleton,run_id,token) VALUES(1,?,?)", (run_id, token))
        connection.commit()
    return token


def release_run(run_id, db_path=None):
    with get_db_connection(db_path) as connection:
        connection.execute("DELETE FROM workspace_lease WHERE run_id=?", (run_id,))
        connection.commit()


def assert_mutation_access(connection, *, resetting=False):
    owner = connection.execute("SELECT run_id,token FROM workspace_lease").fetchone()
    if owner and (resetting or mutation_owner.get() != (owner["run_id"], owner["token"])):
        raise WorkspaceBusy("Workspace mutation blocked while a run owns it")


def recover_interrupted(db_path=None):
    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("UPDATE runs SET status='interrupted',error='Server interrupted before run completed',updated_at=? WHERE status IN ('planning','running','verifying')", (now,))
        connection.execute("DELETE FROM workspace_lease")
        connection.commit()
