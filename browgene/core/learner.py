"""
Learner — Learn mode: Convert AI explorations into deterministic BrowserTasks.
Also supports manual recording sessions.
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from .task import BrowserStep, BrowserTask, StepType
from .explorer import ExplorationResult, RecordedAction

logger = logging.getLogger("browgene.learner")


class Learner:
    """
    Converts exploration sessions into reusable, parameterized BrowserTasks.
    Also supports manual recording where actions are captured live.
    """

    def from_exploration(
        self,
        exploration: ExplorationResult,
        task_name: str,
        description: str = "",
        parameterize: Optional[Dict[str, str]] = None,
    ) -> BrowserTask:
        """
        Convert an ExplorationResult into a deterministic BrowserTask.

        Args:
            exploration: The completed exploration result.
            task_name: Name for the new task.
            description: Task description.
            parameterize: Map of step_param paths to variable names.
                e.g. {"step_3.value": "${policy_number}"}

        Returns:
            A BrowserTask ready for Execute mode.
        """
        logger.info(
            f"LEARN: Converting exploration '{exploration.exploration_id}' "
            f"→ task '{task_name}' ({len(exploration.recorded_actions)} actions)"
        )

        task = BrowserTask(
            name=task_name,
            description=description or exploration.task_description,
            source="explored",
            start_url=(
                exploration.recorded_actions[0].page_url
                if exploration.recorded_actions
                else None
            ),
            metadata={
                "exploration_id": exploration.exploration_id,
                "original_task": exploration.task_description,
            },
        )

        # Convert each recorded action to a BrowserStep
        for i, action in enumerate(exploration.recorded_actions):
            step = action.to_browser_step()
            # Clean up AI-specific descriptions
            if step.description.startswith("AI: "):
                step.description = step.description[4:]
            task.add_step(step)

        # Apply parameterization
        if parameterize:
            task = self._apply_parameters(task, parameterize)

        logger.info(f"LEARN: Created task '{task_name}' with {len(task.steps)} steps")
        return task

    def from_steps(
        self,
        steps: List[Dict[str, Any]],
        task_name: str,
        description: str = "",
        start_url: Optional[str] = None,
        parameters: Optional[Dict[str, str]] = None,
    ) -> BrowserTask:
        """
        Create a BrowserTask from a list of step dictionaries.
        Useful for manual task definition via API.
        """
        task = BrowserTask(
            name=task_name,
            description=description,
            source="manual",
            start_url=start_url,
            parameters=parameters or {},
        )
        for step_data in steps:
            task.add_step(BrowserStep.from_dict(step_data))
        return task

    def optimize_task(self, task: BrowserTask) -> BrowserTask:
        """
        Optimize a task by:
        - Removing redundant waits
        - Merging consecutive type/keyboard actions
        - Removing duplicate screenshots
        - Cleaning up unnecessary navigation
        """
        logger.info(f"OPTIMIZE: Processing task '{task.name}' ({len(task.steps)} steps)")
        optimized_steps: List[BrowserStep] = []

        for i, step in enumerate(task.steps):
            # Skip redundant waits (consecutive waits or very short waits)
            if step.step_type == StepType.WAIT.value:
                seconds = step.params.get("seconds", 0)
                if seconds < 0.2:
                    continue
                # Merge consecutive waits
                if optimized_steps and optimized_steps[-1].step_type == StepType.WAIT.value:
                    prev_seconds = optimized_steps[-1].params.get("seconds", 0)
                    optimized_steps[-1].params["seconds"] = prev_seconds + seconds
                    continue

            # Skip duplicate consecutive screenshots
            if step.step_type == StepType.SCREENSHOT.value:
                if optimized_steps and optimized_steps[-1].step_type == StepType.SCREENSHOT.value:
                    continue

            # Skip navigate to same URL
            if step.step_type == StepType.NAVIGATE.value and i > 0:
                prev_nav = next(
                    (s for s in reversed(optimized_steps) if s.step_type == StepType.NAVIGATE.value),
                    None,
                )
                if prev_nav and prev_nav.params.get("url") == step.params.get("url"):
                    continue

            optimized_steps.append(step)

        removed = len(task.steps) - len(optimized_steps)
        if removed > 0:
            logger.info(f"OPTIMIZE: Removed {removed} redundant steps")

        task.steps = optimized_steps
        task.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        return task

    def _apply_parameters(self, task: BrowserTask, parameterize: Dict[str, str]) -> BrowserTask:
        """
        Replace hardcoded values with ${variable} references.

        parameterize format:
            {"step_3.value": "${policy_number}", "step_5.text": "${username}"}
        """
        for path, variable in parameterize.items():
            parts = path.split(".")
            if len(parts) < 2:
                continue

            # Parse step reference
            step_ref = parts[0]
            param_key = ".".join(parts[1:])

            # Handle "step_N" format
            if step_ref.startswith("step_"):
                try:
                    step_idx = int(step_ref.split("_")[1]) - 1  # 1-indexed to 0-indexed
                except (ValueError, IndexError):
                    continue

                if 0 <= step_idx < len(task.steps):
                    step = task.steps[step_idx]
                    if param_key in step.params:
                        step.params[param_key] = variable
                        logger.info(
                            f"PARAMETERIZE: step_{step_idx + 1}.{param_key} = {variable}"
                        )

                    # Extract variable name for task parameters
                    var_name = variable.strip("${}")
                    if var_name not in task.parameters:
                        task.parameters[var_name] = "string"

        return task


class RecordingSession:
    """
    Captures browser actions in real-time for manual recording.
    User performs actions in the browser; each action is captured as a BrowserStep.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.actions: List[RecordedAction] = []
        self.start_time: str = time.strftime("%Y-%m-%d %H:%M:%S")
        self.end_time: Optional[str] = None
        self.is_recording: bool = False

    def start(self) -> None:
        """Start the recording session."""
        self.is_recording = True
        self.start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"RECORDING [{self.session_id}]: Started")

    def stop(self) -> None:
        """Stop the recording session."""
        self.is_recording = False
        self.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"RECORDING [{self.session_id}]: Stopped ({len(self.actions)} actions)")

    def record_action(self, action: RecordedAction) -> None:
        """Add a recorded action."""
        if self.is_recording:
            self.actions.append(action)
            logger.debug(f"RECORDING: {action.action_type} - {action.description}")

    def to_browser_task(self, task_name: str, description: str = "") -> BrowserTask:
        """Convert the recording to a BrowserTask."""
        task = BrowserTask(
            name=task_name,
            description=description,
            source="recorded",
            start_url=self.actions[0].page_url if self.actions else None,
            metadata={"recording_session_id": self.session_id},
        )
        for action in self.actions:
            task.add_step(action.to_browser_step())
        return task

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the recording session."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "is_recording": self.is_recording,
            "action_count": len(self.actions),
            "actions": [a.to_dict() for a in self.actions],
        }
