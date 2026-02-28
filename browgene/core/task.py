"""
BrowserTask and BrowserStep data models.
Adapted from DeskGene's Task/TaskStep pattern for browser DOM automation.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class StepType(str, Enum):
    """All supported browser step types mapped to Playwright actions."""
    NAVIGATE = "navigate"
    CLICK = "click"
    RIGHT_CLICK = "right_click"
    DOUBLE_CLICK = "double_click"
    FILL = "fill"
    TYPE = "type"
    SELECT = "select"
    SCROLL = "scroll"
    WAIT_FOR = "wait_for"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    EXTRACT = "extract"
    KEYBOARD = "keyboard"
    UPLOAD = "upload"
    EVALUATE = "evaluate"
    AI_ANALYZE = "ai_analyze"
    AI_ACT = "ai_act"
    HOLD_DATA = "hold_data"
    MERGE_DATA = "merge_data"
    ASSERT = "assert"
    HOVER = "hover"
    DRAG_DROP = "drag_drop"
    FRAME_SWITCH = "frame_switch"
    TAB_SWITCH = "tab_switch"
    TAB_OPEN = "tab_open"
    TAB_CLOSE = "tab_close"
    DOWNLOAD = "download"
    DONE = "done"
    SEARCH = "search"


class StepStatus(str, Enum):
    """Runtime status of a step during execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    AI_FALLBACK = "ai_fallback"


class OnFailure(str, Enum):
    """What to do when a step fails."""
    STOP = "stop"
    SKIP = "skip"
    AI_FALLBACK = "ai_fallback"
    RETRY = "retry"


class TaskSource(str, Enum):
    """How the task was created."""
    MANUAL = "manual"
    EXPLORED = "explored"
    RECORDED = "recorded"


class TaskMode(str, Enum):
    """How the task is executed."""
    DETERMINISTIC = "deterministic"   # Step-based replay
    AGENTIC = "agentic"              # AI agent free browsing with a goal


@dataclass
class BrowserStep:
    """
    A single browser automation step.
    Maps directly to Playwright actions with AI fallback support.
    """
    step_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    delay_ms: int = 200
    take_screenshot: bool = False
    on_failure: str = "stop"
    max_retries: int = 1
    timeout_ms: int = 30000
    # Runtime fields (not serialized to task JSON)
    status: str = "pending"
    screenshot_before: Optional[str] = None
    screenshot_after: Optional[str] = None
    extracted_data: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None
    ai_analysis_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize step to dictionary for JSON storage."""
        result: Dict[str, Any] = {
            "step_type": self.step_type,
            "params": self.params,
            "description": self.description,
            "delay_ms": self.delay_ms,
            "take_screenshot": self.take_screenshot,
            "on_failure": self.on_failure,
            "max_retries": self.max_retries,
            "timeout_ms": self.timeout_ms,
        }
        # Include runtime data only if present
        if self.screenshot_before:
            result["screenshot_before"] = self.screenshot_before
        if self.screenshot_after:
            result["screenshot_after"] = self.screenshot_after
        if self.extracted_data is not None:
            result["extracted_data"] = self.extracted_data
        if self.ai_analysis_result is not None:
            result["ai_analysis_result"] = self.ai_analysis_result
        if self.error:
            result["error"] = self.error
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        if self.status != "pending":
            result["status"] = self.status
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrowserStep":
        """Create a BrowserStep from a dictionary."""
        step = cls(
            step_type=data.get("step_type", "navigate"),
            params=data.get("params", {}),
            description=data.get("description", ""),
            delay_ms=data.get("delay_ms", 200),
            take_screenshot=data.get("take_screenshot", False),
            on_failure=data.get("on_failure", "stop"),
            max_retries=data.get("max_retries", 1),
            timeout_ms=data.get("timeout_ms", 30000),
        )
        step.status = data.get("status", "pending")
        step.screenshot_before = data.get("screenshot_before")
        step.screenshot_after = data.get("screenshot_after")
        step.extracted_data = data.get("extracted_data")
        step.ai_analysis_result = data.get("ai_analysis_result")
        step.error = data.get("error")
        step.duration_ms = data.get("duration_ms")
        return step

    def reset(self) -> None:
        """Reset runtime state for re-execution."""
        self.status = StepStatus.PENDING.value
        self.screenshot_before = None
        self.screenshot_after = None
        self.extracted_data = None
        self.ai_analysis_result = None
        self.error = None
        self.duration_ms = None

    def __str__(self) -> str:
        return f"[{self.step_type}] {self.description}"


@dataclass
class BrowserTask:
    """
    A complete browser automation task with ordered steps.
    Adapted from DeskGene's Task pattern for browser DOM.
    """
    name: str
    description: str = ""
    steps: List[BrowserStep] = field(default_factory=list)
    parameters: Dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    updated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    source: str = "manual"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Browser context
    start_url: Optional[str] = None
    requires_auth: bool = False
    auth_credential_key: Optional[str] = None
    # Task mode: deterministic (step-based) or agentic (AI goal-driven)
    mode: str = "deterministic"
    goal: Optional[str] = None           # Natural language goal for agentic tasks
    max_agent_steps: int = 25            # Max steps the AI agent can take

    def add_step(self, step: BrowserStep) -> None:
        """Add a step to the task."""
        self.steps.append(step)
        self.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")

    def insert_step(self, index: int, step: BrowserStep) -> None:
        """Insert a step at a specific position."""
        self.steps.insert(index, step)
        self.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")

    def remove_step(self, index: int) -> Optional[BrowserStep]:
        """Remove a step by index."""
        if 0 <= index < len(self.steps):
            step = self.steps.pop(index)
            self.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
            return step
        return None

    def reset_steps(self) -> None:
        """Reset all steps to pending state."""
        for step in self.steps:
            step.reset()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task to dictionary for JSON storage."""
        d: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
            "parameters": self.parameters,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source": self.source,
            "tags": self.tags,
            "metadata": self.metadata,
            "start_url": self.start_url,
            "requires_auth": self.requires_auth,
            "auth_credential_key": self.auth_credential_key,
            "mode": self.mode,
        }
        if self.goal is not None:
            d["goal"] = self.goal
        if self.max_agent_steps != 25:
            d["max_agent_steps"] = self.max_agent_steps
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrowserTask":
        """Create a BrowserTask from a dictionary."""
        task = cls(
            name=data.get("name", "Unnamed Task"),
            description=data.get("description", ""),
            parameters=data.get("parameters", {}),
            created_at=data.get("created_at", time.strftime("%Y-%m-%d %H:%M:%S")),
            updated_at=data.get("updated_at", time.strftime("%Y-%m-%d %H:%M:%S")),
            source=data.get("source", "manual"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            start_url=data.get("start_url"),
            requires_auth=data.get("requires_auth", False),
            auth_credential_key=data.get("auth_credential_key"),
            mode=data.get("mode", "deterministic"),
            goal=data.get("goal"),
            max_agent_steps=data.get("max_agent_steps", 25),
        )
        task.steps = [
            BrowserStep.from_dict(step_data)
            for step_data in data.get("steps", [])
        ]
        return task

    def to_json(self, indent: int = 2) -> str:
        """Serialize task to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "BrowserTask":
        """Create a BrowserTask from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __str__(self) -> str:
        return f"Task: {self.name} ({len(self.steps)} steps)"


@dataclass
class TaskExecution:
    """
    Tracks the state of a task execution (running, completed, failed).
    Stores memory (hold_data), screenshots, and extracted data.
    """
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    mode: str = "execute"
    status: str = "pending"
    steps_completed: int = 0
    steps_total: int = 0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    memory: Dict[str, Any] = field(default_factory=dict)
    screenshots: List[str] = field(default_factory=list)
    extracted_data: List[Any] = field(default_factory=list)
    error: Optional[str] = None
    # Step-level results
    step_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize execution to dictionary."""
        return {
            "execution_id": self.execution_id,
            "task_name": self.task_name,
            "parameters": self.parameters,
            "mode": self.mode,
            "status": self.status,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "memory": self.memory,
            "screenshots": self.screenshots,
            "extracted_data": self.extracted_data,
            "error": self.error,
            "step_results": self.step_results,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskExecution":
        """Create a TaskExecution from a dictionary."""
        return cls(
            execution_id=data.get("execution_id", str(uuid.uuid4())[:8]),
            task_name=data.get("task_name", ""),
            parameters=data.get("parameters", {}),
            mode=data.get("mode", "execute"),
            status=data.get("status", "pending"),
            steps_completed=data.get("steps_completed", 0),
            steps_total=data.get("steps_total", 0),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            memory=data.get("memory", {}),
            screenshots=data.get("screenshots", []),
            extracted_data=data.get("extracted_data", []),
            error=data.get("error"),
            step_results=data.get("step_results", []),
        )
