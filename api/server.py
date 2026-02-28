"""
BrowGene v2 — FastAPI Server
Exposes all three modes: Explore, Learn, Execute via REST API.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from browgene.core.task import BrowserStep, BrowserTask, TaskExecution
from browgene.core.task_manager import TaskManager
from browgene.core.executor import Executor
from browgene.core.explorer import Explorer
from browgene.core.learner import Learner
from browgene.transport.playwright_driver import PlaywrightDriver
from browgene.ai.vision import VisionAnalyzer
from browgene.ai.fallback import AIFallback

# ── Logging ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("browgene.api")

# ── App Setup ──────────────────────────────────────────────────────────

app = FastAPI(
    title="BrowGene v2",
    description="Three-Mode Browser Automation: Explore → Learn → Execute",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ───────────────────────────────────────────────────────

TASKS_DIR = os.getenv("BROWGENE_TASKS_DIR", "tasks")
SCREENSHOTS_DIR = os.getenv("BROWGENE_SCREENSHOTS_DIR", "executions/screenshots")
HEADLESS = os.getenv("BROWGENE_HEADLESS", "false").lower() == "true"
RECORDINGS_PATH = os.getenv("BROWGENE_RECORDINGS_PATH", "./recordings/browgene-sessions")
SNAPSHOTS_PATH = os.getenv("BROWGENE_SNAPSHOTS_PATH", "./recordings/browgene-snapshots")

# Ensure recording/snapshot directories exist
Path(RECORDINGS_PATH).mkdir(parents=True, exist_ok=True)
Path(SNAPSHOTS_PATH).mkdir(parents=True, exist_ok=True)

task_manager = TaskManager(tasks_dir=TASKS_DIR)
explorer = Explorer(
    llm_provider=os.getenv("BROWGENE_LLM_PROVIDER", "openai"),
    llm_model=os.getenv("BROWGENE_LLM_MODEL", "gpt-4o"),
    headless=HEADLESS,
)
learner = Learner()

# Active driver and execution tracking
_active_driver: Optional[PlaywrightDriver] = None
_executions: Dict[str, TaskExecution] = {}


# ── Request/Response Models ────────────────────────────────────────────

class StepModel(BaseModel):
    step_type: str
    params: Dict[str, Any] = {}
    description: str = ""
    delay_ms: int = 500
    take_screenshot: bool = True
    on_failure: str = "stop"
    max_retries: int = 1
    timeout_ms: int = 30000


class CreateTaskRequest(BaseModel):
    name: str
    description: str = ""
    steps: List[StepModel] = []
    parameters: Dict[str, str] = {}
    start_url: Optional[str] = None
    requires_auth: bool = False
    auth_credential_key: Optional[str] = None
    tags: List[str] = []
    mode: str = "deterministic"      # "deterministic" or "agentic"
    goal: Optional[str] = None        # Natural language goal for agentic tasks
    max_agent_steps: int = 25


class UpdateTaskRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    steps: Optional[List[StepModel]] = None
    parameters: Optional[Dict[str, str]] = None
    start_url: Optional[str] = None
    tags: Optional[List[str]] = None
    mode: Optional[str] = None
    goal: Optional[str] = None
    max_agent_steps: Optional[int] = None


class ExecuteRequest(BaseModel):
    task_name: str
    parameters: Dict[str, Any] = {}
    from_step: int = 0
    headless: Optional[bool] = None
    on_step_failure: str = "stop"


class ExploreRequest(BaseModel):
    task: str
    start_url: Optional[str] = None
    max_steps: int = 25
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None


class LearnFromExplorationRequest(BaseModel):
    exploration_id: str
    task_name: str
    description: str = ""
    parameterize: Optional[Dict[str, str]] = None
    optimize: bool = True


class BrowserLaunchRequest(BaseModel):
    headless: bool = False
    browser_type: str = "chromium"
    viewport_width: int = 1920
    viewport_height: int = 1080
    user_data_dir: Optional[str] = None


# ── Health ─────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "browgene-v2",
        "version": "2.0.0",
        "modes": ["explore", "learn", "execute"],
        "browser_active": _active_driver is not None,
    }


# ── Task Management ───────────────────────────────────────────────────

@app.post("/api/tasks")
async def create_task(req: CreateTaskRequest):
    """Create a new task definition."""
    if task_manager.task_exists(req.name):
        raise HTTPException(400, f"Task '{req.name}' already exists")

    task = BrowserTask(
        name=req.name,
        description=req.description,
        parameters=req.parameters,
        start_url=req.start_url,
        requires_auth=req.requires_auth,
        auth_credential_key=req.auth_credential_key,
        tags=req.tags,
        source="manual",
        mode=req.mode,
        goal=req.goal,
        max_agent_steps=req.max_agent_steps,
    )
    for step_model in req.steps:
        task.add_step(BrowserStep(
            step_type=step_model.step_type,
            params=step_model.params,
            description=step_model.description,
            delay_ms=step_model.delay_ms,
            take_screenshot=step_model.take_screenshot,
            on_failure=step_model.on_failure,
            max_retries=step_model.max_retries,
            timeout_ms=step_model.timeout_ms,
        ))

    filepath = task_manager.save_task(task)
    return {"status": "created", "name": task.name, "mode": task.mode, "steps": len(task.steps), "file": filepath}


@app.get("/api/tasks")
async def list_tasks():
    """List all available tasks."""
    return {"tasks": task_manager.list_tasks()}


@app.get("/api/tasks/{name}")
async def get_task(name: str):
    """Get a task by name."""
    task = task_manager.load_task(name)
    if not task:
        raise HTTPException(404, f"Task '{name}' not found")
    return task.to_dict()


@app.put("/api/tasks/{name}")
async def update_task(name: str, req: UpdateTaskRequest):
    """Update an existing task."""
    task = task_manager.load_task(name)
    if not task:
        raise HTTPException(404, f"Task '{name}' not found")

    if req.name is not None:
        task.name = req.name
    if req.description is not None:
        task.description = req.description
    if req.parameters is not None:
        task.parameters = req.parameters
    if req.start_url is not None:
        task.start_url = req.start_url
    if req.tags is not None:
        task.tags = req.tags
    if req.mode is not None:
        task.mode = req.mode
    if req.goal is not None:
        task.goal = req.goal
    if req.max_agent_steps is not None:
        task.max_agent_steps = req.max_agent_steps
    if req.steps is not None:
        task.steps = [
            BrowserStep(
                step_type=s.step_type,
                params=s.params,
                description=s.description,
                delay_ms=s.delay_ms,
                take_screenshot=s.take_screenshot,
                on_failure=s.on_failure,
                max_retries=s.max_retries,
                timeout_ms=s.timeout_ms,
            )
            for s in req.steps
        ]

    task.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    task_manager.update_task(name, task)
    return {"status": "updated", "name": task.name}


@app.delete("/api/tasks/{name}")
async def delete_task(name: str):
    """Delete a task."""
    if not task_manager.delete_task(name):
        raise HTTPException(404, f"Task '{name}' not found")
    return {"status": "deleted", "name": name}


# ── Execute Mode ───────────────────────────────────────────────────────

@app.post("/api/execute")
async def execute_task(req: ExecuteRequest, background_tasks: BackgroundTasks):
    """Execute a task — deterministic (step replay) or agentic (AI agent)."""
    task = task_manager.load_task(req.task_name)
    if not task:
        raise HTTPException(404, f"Task '{req.task_name}' not found")

    # ── Agentic mode: AI agent free browsing (async) ──────────────────
    if task.mode == "agentic":
        return _start_agentic(task, req)

    # ── Deterministic mode: step replay ───────────────────────────────
    task.reset_steps()

    global _active_driver
    headless = req.headless if req.headless is not None else HEADLESS
    if not _active_driver:
        _active_driver = PlaywrightDriver(
            headless=headless,
            screenshots_dir=SCREENSHOTS_DIR,
        )
        await _active_driver.launch()

    ai_fallback_fn = None
    if req.on_step_failure == "ai_fallback":
        vision = VisionAnalyzer()
        ai_fallback = AIFallback(vision=vision, use_browser_use=True)
        ai_fallback_fn = ai_fallback.handle_failure

    executor = Executor(driver=_active_driver, ai_fallback_fn=ai_fallback_fn)

    execution = await executor.execute_task(
        task=task,
        parameters=req.parameters,
        from_step=req.from_step,
    )

    # Close browser after execution so it doesn't cover the UI
    try:
        await _active_driver.close()
    except Exception as e:
        logger.debug(f"Browser close error (non-fatal): {e}")
    _active_driver = None

    _executions[execution.execution_id] = execution
    return execution.to_dict()


def _start_agentic(task: BrowserTask, req: ExecuteRequest):
    """Launch agentic execution in background. Returns immediately with execution_id."""
    goal = task.goal or task.description
    if not goal:
        raise HTTPException(400, "Agentic task requires a 'goal' or 'description'")

    # Create placeholder execution in "running" state
    execution = TaskExecution(
        task_name=task.name,
        parameters=req.parameters,
        mode="agentic",
        status="running",
        start_time=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    _executions[execution.execution_id] = execution

    # Launch background task
    asyncio.ensure_future(_run_agentic_background(task, req, execution))

    return {
        "execution_id": execution.execution_id,
        "status": "running",
        "mode": "agentic",
        "message": f"Agentic task started. Poll /api/executions/{execution.execution_id} for results.",
    }


async def _run_agentic_background(task: BrowserTask, req: ExecuteRequest, execution: TaskExecution):
    """Background worker for agentic execution."""
    goal = task.goal or task.description
    prompt = goal
    if task.start_url:
        prompt = f"Go to {task.start_url} — then: {goal}"

    exp_id = str(uuid.uuid4())[:8]

    try:
        agent_explorer = Explorer(
            llm_provider=explorer.llm_provider,
            llm_model=explorer.llm_model,
            max_steps=task.max_agent_steps,
            headless=HEADLESS,
            recordings_path=RECORDINGS_PATH,
            snapshots_path=SNAPSHOTS_PATH,
        )

        logger.info(f"Agentic execution of '{task.name}' [{execution.execution_id}]: {prompt} (max {task.max_agent_steps} steps)")

        result = await agent_explorer.explore(
            task=prompt,
            start_url=task.start_url,
            exploration_id=exp_id,
        )

        # Store as exploration for reference
        explorer._explorations[exp_id] = result

        # Update execution with results
        execution.status = "completed" if result.success else "failed"
        execution.steps_completed = len(result.recorded_actions)
        execution.steps_total = len(result.recorded_actions)
        execution.end_time = result.end_time
        execution.error = result.error
        execution.extracted_data = [{
            "agent_output": result.agent_output,
            "final_url": result.final_url,
            "actions_taken": [
                {
                    "action_type": a.action_type,
                    "description": a.description,
                    "page_url": a.page_url,
                }
                for a in result.recorded_actions
            ],
            "steps": result.agent_steps,
            "video_recording_url": result.video_recording_url,
        }]
        execution.memory = {"exploration_id": exp_id}

        if result.final_screenshot:
            execution.screenshots.append(result.final_screenshot)

        # Collect all step screenshot URLs
        for step in result.agent_steps:
            ss_url = step.get("screenshotUrl")
            if ss_url:
                execution.screenshots.append(ss_url)

        logger.info(f"Agentic execution [{execution.execution_id}] completed: {execution.status}")

    except Exception as e:
        logger.error(f"Agentic execution [{execution.execution_id}] failed: {e}")
        execution.status = "failed"
        execution.error = str(e)
        execution.end_time = time.strftime("%Y-%m-%d %H:%M:%S")


@app.post("/api/execute/{execution_id}/resume")
async def resume_execution(execution_id: str):
    """Resume a failed execution from the last completed step."""
    if execution_id not in _executions:
        raise HTTPException(404, f"Execution '{execution_id}' not found")

    prev = _executions[execution_id]
    if prev.status != "failed":
        raise HTTPException(400, f"Execution is not in failed state (status={prev.status})")

    task = task_manager.load_task(prev.task_name)
    if not task:
        raise HTTPException(404, f"Task '{prev.task_name}' not found")

    global _active_driver
    if not _active_driver:
        _active_driver = PlaywrightDriver(headless=HEADLESS, screenshots_dir=SCREENSHOTS_DIR)
        await _active_driver.launch()

    executor = Executor(driver=_active_driver)
    execution = await executor.execute_task(
        task=task,
        parameters=prev.parameters,
        from_step=prev.steps_completed,
    )
    _executions[execution.execution_id] = execution
    return execution.to_dict()


@app.get("/api/executions")
async def list_executions():
    """List all executions."""
    return {
        "executions": [
            {
                "execution_id": e.execution_id,
                "task_name": e.task_name,
                "status": e.status,
                "steps_completed": e.steps_completed,
                "steps_total": e.steps_total,
                "mode": e.mode,
                "start_time": e.start_time,
                "end_time": e.end_time,
            }
            for e in _executions.values()
        ]
    }


@app.get("/api/executions/{execution_id}")
async def get_execution(execution_id: str):
    """Get execution details."""
    if execution_id not in _executions:
        raise HTTPException(404, f"Execution '{execution_id}' not found")
    return _executions[execution_id].to_dict()


# ── Explore Mode ───────────────────────────────────────────────────────

@app.post("/api/explore")
async def start_exploration(req: ExploreRequest, background_tasks: BackgroundTasks):
    """Start an AI exploration session in the background. Returns immediately with exploration_id."""
    exp_id = str(uuid.uuid4())[:8]

    exp_explorer = Explorer(
        llm_provider=req.llm_provider or explorer.llm_provider,
        llm_model=req.llm_model or explorer.llm_model,
        max_steps=req.max_steps,
        headless=HEADLESS,
        recordings_path=RECORDINGS_PATH,
        snapshots_path=SNAPSHOTS_PATH,
    )

    # Pre-register a "running" placeholder so polling doesn't 404
    from browgene.core.explorer import ExplorationResult
    placeholder = ExplorationResult(exploration_id=exp_id, task_description=req.task)
    explorer._explorations[exp_id] = placeholder

    async def _run_exploration():
        result = await exp_explorer.explore(
            task=req.task,
            start_url=req.start_url,
            exploration_id=exp_id,
        )
        # Update the global explorer with the final result
        explorer._explorations[exp_id] = result

    asyncio.ensure_future(_run_exploration())

    return {
        "exploration_id": exp_id,
        "status": "running",
        "message": "Exploration started. Poll /api/explorations/{id} for progress.",
    }


@app.get("/api/explorations")
async def list_explorations():
    """List all completed explorations."""
    return {"explorations": explorer.list_explorations()}


@app.get("/api/explorations/{exploration_id}")
async def get_exploration(exploration_id: str):
    """Get exploration details."""
    result = explorer.get_exploration(exploration_id)
    if not result:
        raise HTTPException(404, f"Exploration '{exploration_id}' not found")
    return result.to_dict()


# ── Learn Mode ─────────────────────────────────────────────────────────

@app.post("/api/learn/from-exploration")
async def learn_from_exploration(req: LearnFromExplorationRequest):
    """Convert an exploration into a deterministic task."""
    exploration = explorer.get_exploration(req.exploration_id)
    if not exploration:
        raise HTTPException(404, f"Exploration '{req.exploration_id}' not found")

    task = learner.from_exploration(
        exploration=exploration,
        task_name=req.task_name,
        description=req.description,
        parameterize=req.parameterize,
    )

    if req.optimize:
        task = learner.optimize_task(task)

    filepath = task_manager.save_task(task)
    return {
        "status": "created",
        "name": task.name,
        "steps": len(task.steps),
        "source": task.source,
        "file": filepath,
    }


# ── Browser Management ─────────────────────────────────────────────────

@app.post("/api/browser/launch")
async def launch_browser(req: BrowserLaunchRequest):
    """Launch a browser session."""
    global _active_driver
    if _active_driver:
        raise HTTPException(400, "Browser already active. Close it first.")

    _active_driver = PlaywrightDriver(
        headless=req.headless,
        browser_type=req.browser_type,
        viewport_width=req.viewport_width,
        viewport_height=req.viewport_height,
        user_data_dir=req.user_data_dir,
        screenshots_dir=SCREENSHOTS_DIR,
    )
    await _active_driver.launch()
    return {"status": "launched", "browser_type": req.browser_type, "headless": req.headless}


@app.post("/api/browser/close")
async def close_browser():
    """Close the active browser session."""
    global _active_driver
    if not _active_driver:
        raise HTTPException(400, "No active browser session")

    await _active_driver.close()
    _active_driver = None
    return {"status": "closed"}


@app.get("/api/browser/status")
async def browser_status():
    """Get browser session status."""
    if not _active_driver:
        return {"active": False}
    return {
        "active": True,
        "url": _active_driver.current_url() if _active_driver._page else None,
    }


@app.post("/api/browser/screenshot")
async def take_screenshot(label: str = "manual"):
    """Take a screenshot of the current page."""
    if not _active_driver:
        raise HTTPException(400, "No active browser session")
    path = await _active_driver.screenshot(label=label)
    return {"status": "captured", "path": path}


@app.post("/api/browser/navigate")
async def navigate_browser(url: str):
    """Navigate the browser to a URL."""
    if not _active_driver:
        raise HTTPException(400, "No active browser session")
    await _active_driver.navigate(url)
    return {"status": "navigated", "url": url}


# ── Memory ─────────────────────────────────────────────────────────────

@app.get("/api/memory/{execution_id}")
async def get_execution_memory(execution_id: str):
    """Get stored data from an execution."""
    if execution_id not in _executions:
        raise HTTPException(404, f"Execution '{execution_id}' not found")
    return {"memory": _executions[execution_id].memory}


# ── File Serving: Recordings & Snapshots ──────────────────────────────

@app.get("/api/browgene/recordings/{filename}")
async def serve_recording(filename: str):
    """Serve a video recording file."""
    filepath = Path(RECORDINGS_PATH) / filename
    if not filepath.exists():
        raise HTTPException(404, f"Recording '{filename}' not found")
    media_type = "video/webm" if filename.endswith(".webm") else "video/mp4"
    return FileResponse(str(filepath), media_type=media_type, filename=filename)


@app.get("/api/browgene/snapshots/{filename}")
async def serve_snapshot(filename: str):
    """Serve a snapshot screenshot file."""
    filepath = Path(SNAPSHOTS_PATH) / filename
    if not filepath.exists():
        raise HTTPException(404, f"Snapshot '{filename}' not found")
    media_type = "image/png" if filename.endswith(".png") else "image/jpeg"
    return FileResponse(str(filepath), media_type=media_type, filename=filename)


# ── Browser-Use Cloud API v2 Compatible Endpoints ─────────────────────
# These endpoints mirror the browser-use Cloud v2 API so PulseGene's
# brow-gene.ts node can use BrowGene interchangeably by just changing
# the base URL from https://api.browser-use.com/api/v2 to
# http://localhost:8200/api/v2

# v2 task storage: task_id -> {status, output, steps, exploration, ...}
_v2_tasks: Dict[str, Dict[str, Any]] = {}


class V2CreateTaskRequest(BaseModel):
    task: str
    maxSteps: int = 25
    enableRecording: bool = False
    keepAlive: bool = False
    structuredOutput: Optional[Dict[str, Any]] = None
    systemPromptExtension: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@app.post("/api/v2/tasks")
async def v2_create_task(req: V2CreateTaskRequest, request: Request):
    """Create a browser-use task (v2 compatible). Returns immediately with task ID."""
    task_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    # Store initial task state
    _v2_tasks[task_id] = {
        "id": task_id,
        "sessionId": session_id,
        "status": "running",
        "output": None,
        "result": None,
        "structured_output": None,
        "steps": [],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_description": req.task,
        "max_steps": req.maxSteps,
        "exploration_id": None,
    }

    # Launch exploration in background
    asyncio.ensure_future(_run_v2_task(task_id, req))

    logger.info(f"v2 task created: {task_id} — {req.task[:80]}")
    return {
        "id": task_id,
        "sessionId": session_id,
        "status": "running",
    }


async def _run_v2_task(task_id: str, req: V2CreateTaskRequest):
    """Background worker for v2-compatible task execution."""
    task_state = _v2_tasks.get(task_id)
    if not task_state:
        return

    exp_id = str(uuid.uuid4())[:8]
    task_state["exploration_id"] = exp_id

    try:
        max_steps = req.maxSteps or 25
        agent_explorer = Explorer(
            llm_provider=explorer.llm_provider,
            llm_model=explorer.llm_model,
            max_steps=max_steps,
            headless=HEADLESS,
            recordings_path=RECORDINGS_PATH if req.enableRecording else None,
            snapshots_path=SNAPSHOTS_PATH,
        )

        logger.info(f"v2 task [{task_id}] exploring: {req.task[:80]} (max {max_steps} steps, recording={req.enableRecording})")

        result = await agent_explorer.explore(
            task=req.task,
            exploration_id=exp_id,
        )

        # Store exploration for cross-reference
        explorer._explorations[exp_id] = result

        # Map status to v2 format
        v2_status = "finished" if result.success else "failed"

        # Build v2-compatible steps from agent_steps
        v2_steps = result.agent_steps if result.agent_steps else []

        # Update task state
        task_state["status"] = v2_status
        task_state["output"] = result.agent_output or ""
        task_state["result"] = result.agent_output or ""
        task_state["video_recording_url"] = result.video_recording_url
        task_state["steps"] = v2_steps
        task_state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        logger.info(f"v2 task [{task_id}] {v2_status}: {len(v2_steps)} steps, output={len(result.agent_output or '')} chars")

    except Exception as e:
        logger.error(f"v2 task [{task_id}] failed: {e}")
        task_state["status"] = "failed"
        task_state["output"] = str(e)


@app.get("/api/v2/tasks/{task_id}")
async def v2_get_task(task_id: str):
    """Get task status (v2 compatible). Same shape as browser-use Cloud polling."""
    if task_id not in _v2_tasks:
        raise HTTPException(404, f"Task '{task_id}' not found")

    task_state = _v2_tasks[task_id]

    # If still running, also check exploration progress for live steps
    if task_state["status"] == "running" and task_state.get("exploration_id"):
        exp = explorer.get_exploration(task_state["exploration_id"])
        if exp and exp.agent_steps:
            task_state["steps"] = exp.agent_steps

    # Collect screenshot URLs from steps
    screenshot_urls = [
        s.get("screenshotUrl") for s in task_state.get("steps", [])
        if s.get("screenshotUrl")
    ]

    return {
        "id": task_state["id"],
        "sessionId": task_state["sessionId"],
        "status": task_state["status"],
        "output": task_state.get("output"),
        "result": task_state.get("result"),
        "structured_output": task_state.get("structured_output"),
        "structuredOutput": task_state.get("structured_output"),
        "steps": task_state.get("steps", []),
        "videoRecordingUrl": task_state.get("video_recording_url"),
        "screenshots": screenshot_urls,
    }


@app.delete("/api/v2/tasks/{task_id}")
async def v2_stop_task(task_id: str):
    """Stop/delete a task (v2 compatible)."""
    if task_id not in _v2_tasks:
        raise HTTPException(404, f"Task '{task_id}' not found")
    logger.info(f"v2 task [{task_id}] stop requested")
    return {"id": task_id, "status": "stopped"}


@app.get("/api/v2/tasks/{task_id}/files")
async def v2_get_task_files(task_id: str):
    """Get task output files (v2 compatible). Returns empty list for local execution."""
    if task_id not in _v2_tasks:
        raise HTTPException(404, f"Task '{task_id}' not found")
    # Local execution doesn't produce cloud-hosted files
    return []


@app.get("/api/v2/sessions/{session_id}")
async def v2_get_session(session_id: str):
    """Get session details (v2 compatible). Returns liveUrl for the session."""
    # Find the task that owns this session
    for task_state in _v2_tasks.values():
        if task_state.get("sessionId") == session_id:
            return {
                "id": session_id,
                "liveUrl": "",  # No live URL for local headful execution
                "status": "active" if task_state["status"] == "running" else "closed",
            }
    raise HTTPException(404, f"Session '{session_id}' not found")


# ── Startup/Shutdown ───────────────────────────────────────────────────

@app.on_event("shutdown")
async def shutdown():
    global _active_driver
    if _active_driver:
        await _active_driver.close()
        _active_driver = None
    logger.info("BrowGene v2 server shutting down")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("BROWGENE_PORT", "8200"))
    uvicorn.run(app, host="0.0.0.0", port=port)
