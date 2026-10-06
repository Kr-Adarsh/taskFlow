"""
Main FastAPI application entry point for Operon.
Mounts workspace endpoints, agent run control, and SSE streams.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.app.workspace.db import init_db
from backend.app.workspace.seed import seed_workspace
from backend.app.workspace.lease import recover_interrupted, mutation_owner, WorkspaceBusy, assert_mutation_access
from backend.app.tools.browser_tools import browser_manager
from backend.app.api.workspace import router as workspace_router
from backend.app.api.runs import router as runs_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize workspace DB and seed baseline data on startup
    init_db()
    recover_interrupted()
    seed_workspace()
    try:
        yield
    finally:
        await browser_manager.close()

app = FastAPI(
    title="Operon Autonomous AI Worker",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(workspace_router)
app.include_router(runs_router)

from pathlib import Path
from fastapi.responses import RedirectResponse, HTMLResponse

DASHBOARD_FILE = Path(__file__).resolve().parent / "static" / "dashboard.html"

@app.get("/health")
def health_check():
    return {"status": "ok", "product": "Operon", "version": "2.0.0"}

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(content=DASHBOARD_FILE.read_text(encoding="utf-8"))

@app.get("/")
def root():
    return RedirectResponse(url="/dashboard")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)


@app.middleware("http")
async def mutation_context(request, call_next):
    owner = mutation_owner.set((request.headers.get("x-operon-run"), request.headers.get("x-operon-lease")))
    try:
        if request.method == "POST" and request.url.path.startswith(("/api/workspace/", "/workspace/")):
            from backend.app.workspace.db import get_db_connection
            from fastapi.responses import JSONResponse
            try:
                with get_db_connection() as connection:
                    assert_mutation_access(connection, resetting=request.url.path.endswith(("/reset", "/fault-injection")))
            except WorkspaceBusy as error:
                return JSONResponse(status_code=409, content={"ok": False, "error": str(error), "error_code": "WORKSPACE_BUSY"})
        return await call_next(request)
    finally:
        mutation_owner.reset(owner)

@app.exception_handler(WorkspaceBusy)
async def workspace_busy(request, error):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=409, content={"ok": False, "error": str(error), "error_code": "WORKSPACE_BUSY"})
