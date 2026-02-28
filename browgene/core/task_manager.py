"""
TaskManager — Save, load, list, and manage BrowserTask definitions.
Adapted from DeskGene's TaskManager for browser automation.
"""

import glob
import json
import logging
import os
from typing import Dict, List, Optional

from .task import BrowserTask

logger = logging.getLogger("browgene.task_manager")


class TaskManager:
    """Manages BrowserTask persistence — save, load, list, delete, update."""

    def __init__(self, tasks_dir: str = "tasks"):
        self.tasks_dir = tasks_dir
        os.makedirs(tasks_dir, exist_ok=True)

    def _task_filename(self, name: str) -> str:
        """Convert task name to a safe filename."""
        safe_name = name.replace(" ", "_").lower()
        return os.path.join(self.tasks_dir, f"{safe_name}.json")

    def save_task(self, task: BrowserTask) -> str:
        """Save a task to disk. Returns the file path."""
        filepath = self._task_filename(task.name)
        with open(filepath, "w") as f:
            json.dump(task.to_dict(), f, indent=2)
        logger.info(f"Task saved: {task.name} → {filepath}")
        return filepath

    def load_task(self, name: str) -> Optional[BrowserTask]:
        """Load a task by name. Returns None if not found."""
        filepath = self._task_filename(name)
        if not os.path.exists(filepath):
            # Try searching by actual task name in JSON
            return self._find_task_by_name(name)
        return self._load_from_file(filepath)

    def _load_from_file(self, filepath: str) -> Optional[BrowserTask]:
        """Load a task from a specific file path."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            return BrowserTask.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading task from {filepath}: {e}")
            return None

    def _find_task_by_name(self, name: str) -> Optional[BrowserTask]:
        """Search through all task files to find one matching the given name."""
        task_files = glob.glob(os.path.join(self.tasks_dir, "*.json"))
        for filepath in task_files:
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                if data.get("name") == name:
                    return BrowserTask.from_dict(data)
            except Exception:
                continue
        logger.warning(f"Task '{name}' not found")
        return None

    def list_tasks(self) -> List[Dict[str, str]]:
        """List all available tasks with name, description, and step count."""
        tasks: List[Dict[str, str]] = []
        if not os.path.exists(self.tasks_dir):
            return tasks

        task_files = sorted(glob.glob(os.path.join(self.tasks_dir, "*.json")))
        for filepath in task_files:
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                tasks.append({
                    "name": data.get("name", os.path.basename(filepath)),
                    "description": data.get("description", ""),
                    "steps": str(len(data.get("steps", []))),
                    "source": data.get("source", "manual"),
                    "tags": ",".join(data.get("tags", [])),
                    "start_url": data.get("start_url", ""),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "mode": data.get("mode", "deterministic"),
                    "goal": data.get("goal", ""),
                })
            except Exception as e:
                logger.error(f"Error reading task file {filepath}: {e}")
        return tasks

    def delete_task(self, name: str) -> bool:
        """Delete a task by name."""
        filepath = self._task_filename(name)
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Task deleted: {name}")
            return True
        logger.warning(f"Task '{name}' not found for deletion")
        return False

    def update_task(self, old_name: str, updated_task: BrowserTask) -> bool:
        """Update an existing task. If name changed, removes old file."""
        try:
            # Remove old file if name changed
            if old_name != updated_task.name:
                old_filepath = self._task_filename(old_name)
                if os.path.exists(old_filepath):
                    os.remove(old_filepath)

            self.save_task(updated_task)
            logger.info(f"Task updated: {old_name} → {updated_task.name}")
            return True
        except Exception as e:
            logger.error(f"Error updating task: {e}")
            return False

    def task_exists(self, name: str) -> bool:
        """Check if a task exists."""
        filepath = self._task_filename(name)
        return os.path.exists(filepath)

    def duplicate_task(self, name: str, new_name: str) -> Optional[BrowserTask]:
        """Duplicate a task with a new name."""
        task = self.load_task(name)
        if not task:
            return None
        task.name = new_name
        task.source = "manual"
        self.save_task(task)
        return task
