"""
Explorer — Explore mode using browser-use AI agent.
Wraps browser-use to discover unknown page workflows.
Every action is recorded for later conversion to deterministic steps via Learn mode.
"""

import asyncio
import base64
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .task import BrowserStep, BrowserTask, StepType

logger = logging.getLogger("browgene.explorer")


class RecordedAction:
    """Captures a single action performed by the browser-use agent."""

    def __init__(
        self,
        action_type: str,
        params: Dict[str, Any],
        description: str = "",
        screenshot_before: Optional[str] = None,
        screenshot_after: Optional[str] = None,
        page_url: str = "",
        timestamp: Optional[str] = None,
    ):
        self.action_type = action_type
        self.params = params
        self.description = description
        self.screenshot_before = screenshot_before
        self.screenshot_after = screenshot_after
        self.page_url = page_url
        self.timestamp = timestamp or time.strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "params": self.params,
            "description": self.description,
            "screenshot_before": self.screenshot_before,
            "screenshot_after": self.screenshot_after,
            "page_url": self.page_url,
            "timestamp": self.timestamp,
        }

    def to_browser_step(self) -> BrowserStep:
        """Convert this recorded action to a BrowserStep."""
        return BrowserStep(
            step_type=self.action_type,
            params=self.params,
            description=self.description,
            delay_ms=200,
            take_screenshot=False,
            on_failure="ai_fallback",
        )


class ExplorationResult:
    """Result of an exploration session."""

    def __init__(self, exploration_id: str, task_description: str):
        self.exploration_id = exploration_id
        self.task_description = task_description
        self.recorded_actions: List[RecordedAction] = []
        self.final_url: str = ""
        self.final_screenshot: Optional[str] = None
        self.success: bool = False
        self.error: Optional[str] = None
        self.status: str = "running"  # running | completed | failed
        self.start_time: str = time.strftime("%Y-%m-%d %H:%M:%S")
        self.end_time: Optional[str] = None
        self.agent_output: Optional[str] = None
        self.video_recording_url: Optional[str] = None
        # browser-use v2 compatible step data (memory, goals, actions per step)
        self.agent_steps: List[Dict[str, Any]] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exploration_id": self.exploration_id,
            "task_description": self.task_description,
            "recorded_actions": [a.to_dict() for a in self.recorded_actions],
            "final_url": self.final_url,
            "final_screenshot": self.final_screenshot,
            "success": self.success,
            "error": self.error,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "agent_output": self.agent_output,
            "action_count": len(self.recorded_actions),
            "agent_steps": self.agent_steps,
            "video_recording_url": self.video_recording_url,
        }

    def to_browser_task(self, task_name: str) -> BrowserTask:
        """Convert exploration to a BrowserTask (for Learn mode)."""
        task = BrowserTask(
            name=task_name,
            description=self.task_description,
            source="explored",
            start_url=self.recorded_actions[0].page_url if self.recorded_actions else None,
        )
        for action in self.recorded_actions:
            task.add_step(action.to_browser_step())
        return task


class Explorer:
    """
    Explore mode — uses browser-use AI agent to navigate unknown pages.
    Records every action for later conversion to deterministic steps.

    Usage:
        explorer = Explorer(llm_model="gpt-4o")
        result = await explorer.explore(
            task="Find the policy search page and search for POL-12345",
            start_url="https://portal.example.com",
        )
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o",
        max_steps: int = 25,
        headless: bool = False,
        recordings_path: Optional[str] = None,
        snapshots_path: Optional[str] = None,
        # Vertex AI config (used when llm_provider="google")
        use_vertexai: bool = False,
        vertexai_project: Optional[str] = None,
        vertexai_location: str = "us-central1",
    ):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.max_steps = max_steps
        self.headless = headless
        self.recordings_path = recordings_path
        self.snapshots_path = snapshots_path
        self.use_vertexai = use_vertexai
        self.vertexai_project = vertexai_project
        self.vertexai_location = vertexai_location
        self._explorations: Dict[str, ExplorationResult] = {}
        # Active browser sessions keyed by exploration_id for live screenshot capture
        self._active_sessions: Dict[str, Any] = {}

    async def explore(
        self,
        task: str,
        start_url: Optional[str] = None,
        exploration_id: Optional[str] = None,
    ) -> ExplorationResult:
        """
        Run an AI exploration session.

        Args:
            task: Natural language description of what to accomplish.
            start_url: URL to start from (optional).
            exploration_id: Custom ID for this exploration.

        Returns:
            ExplorationResult with recorded actions.
        """
        import uuid

        exp_id = exploration_id or str(uuid.uuid4())[:8]
        result = ExplorationResult(exploration_id=exp_id, task_description=task)

        logger.info(f"═══ EXPLORE [{exp_id}]: {task} ═══")

        # Store result early so polling can see "running" status
        self._explorations[exp_id] = result

        try:
            # Try to import browser-use (v0.12+ API)
            from browser_use import Agent
            from browser_use.browser.profile import BrowserProfile
            from browser_use.browser.session import BrowserSession

            llm = self._create_llm()
            if not llm:
                result.error = "Failed to create LLM client"
                result.status = "failed"
                return result

            # Configure browser session with disable_security for SSL issues
            profile_kwargs: Dict[str, Any] = {
                "headless": self.headless,
                "disable_security": True,
            }

            # Enable video recording if recordings_path is set
            if self.recordings_path:
                rec_dir = Path(self.recordings_path)
                rec_dir.mkdir(parents=True, exist_ok=True)
                profile_kwargs["record_video_dir"] = str(rec_dir)
                logger.info(f"Video recording enabled → {rec_dir}")

            browser_profile = BrowserProfile(**profile_kwargs)
            browser_session = BrowserSession(browser_profile=browser_profile)

            # Store active session for live screenshot capture
            self._active_sessions[exp_id] = browser_session

            # Embed start_url in task text so browser-use's built-in
            # directly_open_url mechanism navigates there automatically.
            # This is the most reliable approach in browser-use 0.12.
            full_task = task
            if start_url:
                full_task = f"Go to {start_url} — then: {task}"
                logger.info(f"Task with URL: {full_task}")

            # Create agent
            agent = Agent(
                task=full_task,
                llm=llm,
                browser=browser_session,
                max_steps=self.max_steps,
            )

            # Run the agent
            agent_result = await agent.run()

            # Extract recorded actions and v2-compatible step data from agent history
            if hasattr(agent_result, 'history') and agent_result.history:
                for step_idx, entry in enumerate(agent_result.history):
                    actions = self._history_entry_to_actions(entry)
                    result.recorded_actions.extend(actions)
                    # Build v2-compatible step object
                    step_data = self._history_entry_to_v2_step(entry, step_idx)
                    if step_data:
                        result.agent_steps.append(step_data)

            result.success = True
            result.status = "completed"
            # Extract clean final text from the agent result
            result.agent_output = self._extract_final_output(agent_result)

            # Save per-step screenshots to disk
            if self.snapshots_path and hasattr(agent_result, 'history') and agent_result.history:
                self._save_step_screenshots(result, agent_result.history, exp_id)

            # Stop browser session
            try:
                await browser_session.stop()
            except Exception as stop_err:
                logger.debug(f"Browser stop error (non-fatal): {stop_err}")
            finally:
                self._active_sessions.pop(exp_id, None)

            # Locate video recording file
            if self.recordings_path:
                self._attach_video_recording(result, exp_id)

        except ImportError as ie:
            logger.warning(f"browser-use not installed or import error: {ie}. Install with: pip install browser-use")
            result.error = f"browser-use import error: {ie}"
            result.status = "failed"
            self._active_sessions.pop(exp_id, None)
        except Exception as e:
            logger.error(f"Exploration failed: {e}")
            result.error = str(e)
            result.status = "failed"
            self._active_sessions.pop(exp_id, None)

        result.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"═══ EXPLORE [{exp_id}] done: {len(result.recorded_actions)} actions recorded ═══")
        return result

    async def get_live_screenshot(self, exploration_id: str) -> Optional[bytes]:
        """Capture a live screenshot from the active browser session.

        Returns PNG bytes if a page is open, or None.
        """
        session = self._active_sessions.get(exploration_id)
        if not session:
            logger.debug(f"No active session for exploration {exploration_id}")
            return None

        try:
            # Preferred: BrowserSession.take_screenshot() returns bytes directly
            # Use JPEG for faster streaming (smaller frames)
            if hasattr(session, 'take_screenshot'):
                screenshot_bytes = await session.take_screenshot(format="jpeg", quality=70)
                if screenshot_bytes:
                    return screenshot_bytes

            # Fallback: get_current_page() -> Playwright Page -> screenshot()
            if hasattr(session, 'get_current_page'):
                page = await session.get_current_page()
                if page:
                    screenshot_bytes = await page.screenshot(type="png")
                    return screenshot_bytes

            # Fallback 2: get_pages() -> last page -> screenshot()
            if hasattr(session, 'get_pages'):
                pages = await session.get_pages()
                if pages:
                    screenshot_bytes = await pages[-1].screenshot(type="png")
                    return screenshot_bytes

            logger.debug(f"No active page found for exploration {exploration_id}")
            return None

        except Exception as e:
            logger.debug(f"Could not capture live screenshot for {exploration_id}: {e}")
            return None

    def get_exploration(self, exploration_id: str) -> Optional[ExplorationResult]:
        """Retrieve a completed exploration by ID."""
        return self._explorations.get(exploration_id)

    def list_explorations(self) -> List[Dict[str, Any]]:
        """List all explorations."""
        return [
            {
                "exploration_id": exp.exploration_id,
                "task": exp.task_description,
                "actions": len(exp.recorded_actions),
                "success": exp.success,
                "start_time": exp.start_time,
            }
            for exp in self._explorations.values()
        ]

    def _create_llm(self) -> Any:
        """Create the LLM client for browser-use (using browser-use's native models)."""
        try:
            if self.llm_provider == "openai":
                from browser_use.llm.models import ChatOpenAI
                return ChatOpenAI(model=self.llm_model)
            elif self.llm_provider == "anthropic":
                from browser_use.llm.models import ChatBrowserUse
                return ChatBrowserUse(model=self.llm_model)
            elif self.llm_provider == "google":
                from browser_use.llm.models import ChatGoogle
                kwargs: Dict[str, Any] = {"model": self.llm_model}
                if self.use_vertexai:
                    kwargs["vertexai"] = True
                    if self.vertexai_project:
                        kwargs["project"] = self.vertexai_project
                    if self.vertexai_location:
                        kwargs["location"] = self.vertexai_location
                    logger.info(f"Creating ChatGoogle with Vertex AI: project={self.vertexai_project}, location={self.vertexai_location}, model={self.llm_model}")
                return ChatGoogle(**kwargs)
            else:
                logger.error(f"Unknown LLM provider: {self.llm_provider}")
                return None
        except ImportError as e:
            logger.error(f"LLM provider package not installed: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create LLM: {e}")
            return None

    @staticmethod
    def _to_plain_dict(obj: Any) -> Any:
        """Recursively convert Pydantic models / nested objects to plain JSON-safe dicts."""
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        if hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool)):
            return {k: Explorer._to_plain_dict(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
        if isinstance(obj, dict):
            return {k: Explorer._to_plain_dict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [Explorer._to_plain_dict(i) for i in obj]
        return obj

    @staticmethod
    def _extract_element_info(entry: Any, action_index: int = 0) -> Dict[str, Any]:
        """Extract interacted element info from browser-use history state."""
        info: Dict[str, Any] = {}
        try:
            if hasattr(entry, 'state') and entry.state:
                interacted = getattr(entry.state, 'interacted_element', None)
                if interacted and isinstance(interacted, list):
                    # Get element for this action index
                    idx = min(action_index, len(interacted) - 1)
                    elem = interacted[idx] if idx >= 0 else None
                    if elem is not None:
                        ax_name = getattr(elem, 'ax_name', None)
                        node_name = getattr(elem, 'node_name', None)
                        x_path = getattr(elem, 'x_path', None)
                        attrs = getattr(elem, 'attributes', None)
                        if ax_name:
                            info['element_text'] = ax_name
                        if node_name:
                            info['element_tag'] = node_name.lower()
                        if x_path:
                            info['xpath'] = x_path
                        if attrs and isinstance(attrs, dict):
                            # Keep useful selector attrs
                            for key in ('id', 'name', 'data-testid', 'aria-label', 'placeholder', 'type', 'href'):
                                if key in attrs:
                                    info.setdefault('element_attrs', {})[key] = attrs[key]
        except Exception as e:
            logger.debug(f"Could not extract element info: {e}")
        return info

    def _save_step_screenshots(self, result: ExplorationResult, history: Any, exp_id: str) -> None:
        """Save per-step screenshots from agent history to disk and update agent_steps with URLs."""
        if not self.snapshots_path:
            return

        snap_dir = Path(self.snapshots_path)
        snap_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0
        for step_idx, entry in enumerate(history):
            try:
                state = getattr(entry, 'state', None)
                if not state:
                    continue
                screenshot_b64 = getattr(state, 'screenshot', None)
                if not screenshot_b64 or not isinstance(screenshot_b64, str):
                    continue

                # Decode and save
                filename = f"snapshot-{exp_id}-step{step_idx}.png"
                filepath = snap_dir / filename
                img_data = base64.b64decode(screenshot_b64)
                filepath.write_bytes(img_data)

                # Build API URL for serving
                api_url = f"/api/browgene/snapshots/{filename}"

                # Update the corresponding agent_step with screenshotUrl
                if step_idx < len(result.agent_steps):
                    result.agent_steps[step_idx]["screenshotUrl"] = api_url

                saved_count += 1
            except Exception as e:
                logger.debug(f"Could not save screenshot for step {step_idx}: {e}")

        if saved_count > 0:
            logger.info(f"📸 Saved {saved_count} step screenshots to {snap_dir}")

        # Also save the final screenshot if available from the last state
        if history:
            try:
                last_state = getattr(history[-1], 'state', None)
                if last_state:
                    final_ss = getattr(last_state, 'screenshot', None)
                    if final_ss and isinstance(final_ss, str):
                        result.final_screenshot = f"/api/browgene/snapshots/snapshot-{exp_id}-step{len(history) - 1}.png"
            except Exception:
                pass

    def _attach_video_recording(self, result: ExplorationResult, exp_id: str) -> None:
        """Find the most recent video recording in recordings_path and attach to result."""
        if not self.recordings_path:
            return

        rec_dir = Path(self.recordings_path)
        if not rec_dir.exists():
            return

        try:
            # browser-use creates .webm files in the record_video_dir
            video_files = sorted(
                rec_dir.glob("*.webm"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if not video_files:
                # Also check for .mp4 files
                video_files = sorted(
                    rec_dir.glob("*.mp4"),
                    key=lambda f: f.stat().st_mtime,
                    reverse=True,
                )

            if video_files:
                latest = video_files[0]
                # Rename to include exploration ID for reliable lookup
                new_name = f"browgene-{exp_id}{latest.suffix}"
                new_path = rec_dir / new_name
                try:
                    latest.rename(new_path)
                    logger.info(f"🎬 Video recording: {latest.name} → {new_name}")
                except Exception:
                    new_path = latest
                    new_name = latest.name

                # Store API URL in result
                result.video_recording_url = f"/api/browgene/recordings/{new_name}"
                logger.info(f"🎬 Video recording attached: {result.video_recording_url}")
            else:
                logger.debug("No video recording files found")
        except Exception as e:
            logger.debug(f"Could not attach video recording: {e}")

    @staticmethod
    def _extract_final_output(agent_result: Any) -> Optional[str]:
        """Extract the clean final text output from a browser-use agent result.
        
        browser-use 0.12 returns an AgentHistoryList. The final 'done' action
        contains the human-readable summary text. We prefer that over str(result)
        which dumps raw Python repr.
        """
        try:
            if not agent_result:
                return None

            # Strategy 1: Check for final_result() method (browser-use 0.12+)
            if hasattr(agent_result, 'final_result') and callable(agent_result.final_result):
                final = agent_result.final_result()
                if final and isinstance(final, str) and len(final) > 10:
                    return final

            # Strategy 2: Walk history backwards to find last 'done' action text
            history = getattr(agent_result, 'history', None)
            if history:
                for entry in reversed(list(history)):
                    model_output = getattr(entry, 'model_output', None)
                    if not model_output:
                        continue
                    action_list = getattr(model_output, 'action', None)
                    if not action_list:
                        continue
                    if not isinstance(action_list, list):
                        action_list = [action_list]
                    for action_obj in action_list:
                        done_data = getattr(action_obj, 'done', None)
                        if done_data is None and isinstance(action_obj, dict):
                            done_data = action_obj.get('done')
                        if done_data:
                            text = getattr(done_data, 'text', None)
                            if text is None and isinstance(done_data, dict):
                                text = done_data.get('text')
                            if text and isinstance(text, str) and len(text) > 5:
                                return text

            # Strategy 3: Check all_results for done text
            all_results = getattr(agent_result, 'all_results', None)
            if all_results:
                for r in reversed(all_results):
                    is_done = getattr(r, 'is_done', False)
                    if is_done:
                        extracted = getattr(r, 'extracted_content', None)
                        if extracted and isinstance(extracted, str) and len(extracted) > 10:
                            return extracted

            # Fallback: str(result) but truncated
            raw = str(agent_result)
            if len(raw) > 2000:
                raw = raw[:2000] + "..."
            return raw
        except Exception as e:
            logger.debug(f"Could not extract final output: {e}")
            return str(agent_result) if agent_result else None

    def _history_entry_to_v2_step(self, entry: Any, step_index: int) -> Optional[Dict[str, Any]]:
        """Convert a browser-use history entry to a browser-use Cloud v2 step object.
        
        Returns a dict matching the v2 API step format:
        {
            "actions": [...],
            "memory": "...",
            "evaluationPreviousGoal": "...",
            "nextGoal": "...",
            "screenshotUrl": "..."
        }
        """
        try:
            step: Dict[str, Any] = {}

            # Extract model output fields (memory, goals)
            model_output = getattr(entry, 'model_output', None)
            if model_output:
                # current_state holds memory/goals in browser-use 0.12
                current_state = getattr(model_output, 'current_state', None)
                if current_state:
                    step["memory"] = getattr(current_state, 'memory', '') or ''
                    step["evaluationPreviousGoal"] = getattr(current_state, 'evaluation_previous_goal', '') or ''
                    step["nextGoal"] = getattr(current_state, 'next_goal', '') or ''
                else:
                    step["memory"] = getattr(model_output, 'memory', '') or ''
                    step["evaluationPreviousGoal"] = getattr(model_output, 'evaluation_previous_goal', '') or ''
                    step["nextGoal"] = getattr(model_output, 'next_goal', '') or ''

                # Extract actions list
                action_list = getattr(model_output, 'action', None)
                if action_list:
                    if not isinstance(action_list, list):
                        action_list = [action_list]
                    step["actions"] = [self._to_plain_dict(a) for a in action_list]
                else:
                    step["actions"] = []
            else:
                step["memory"] = ""
                step["evaluationPreviousGoal"] = ""
                step["nextGoal"] = ""
                step["actions"] = []

            # Extract result/extracted_content from step results
            results = getattr(entry, 'result', None)
            if results:
                if isinstance(results, list):
                    for r in results:
                        extracted = getattr(r, 'extracted_content', None)
                        if extracted:
                            step["output"] = str(extracted)
                            break
                else:
                    extracted = getattr(results, 'extracted_content', None)
                    if extracted:
                        step["output"] = str(extracted)

            # Extract screenshot from state
            state = getattr(entry, 'state', None)
            if state:
                screenshot = getattr(state, 'screenshot', None)
                if screenshot and isinstance(screenshot, str):
                    # If it's a base64 string, we could serve it as a URL later
                    step["screenshotUrl"] = None  # Local screenshots handled separately
                url = getattr(state, 'url', None)
                if url:
                    step["url"] = url

            return step
        except Exception as e:
            logger.debug(f"Could not convert history entry to v2 step: {e}")
            return {"actions": [], "memory": "", "evaluationPreviousGoal": "", "nextGoal": ""}

    def _history_entry_to_actions(self, entry: Any) -> List[RecordedAction]:
        """Convert a browser-use history entry to a list of RecordedActions.
        
        Browser-use can execute multiple actions per step (e.g. click Auto-Fill + click Sign In).
        Each action in the list must be captured separately.
        """
        results: List[RecordedAction] = []
        try:
            if not (hasattr(entry, 'model_output') and entry.model_output):
                return results

            action_list = entry.model_output.action
            if not action_list:
                return results

            # Normalize to a list of action objects
            if not isinstance(action_list, list):
                action_list = [action_list]

            for action_idx, action_obj in enumerate(action_list):
                action_dict = self._to_plain_dict(action_obj)

                if not isinstance(action_dict, dict):
                    results.append(RecordedAction(
                        action_type="ai_act",
                        params={"raw": str(action_dict)},
                        description=str(action_dict),
                    ))
                    continue

                # Extract element info for this specific action
                elem_info = self._extract_element_info(entry, action_idx)

                for action_name, action_params in action_dict.items():
                    if action_params is None:
                        continue
                    if not isinstance(action_params, dict):
                        action_params = {"value": action_params}
                    mapped = self._map_browser_use_action(action_name, action_params, elem_info)
                    if mapped:
                        results.append(mapped)
                        break  # Each action_obj has one action key

        except Exception as e:
            logger.debug(f"Could not convert history entry: {e}")
        return results

    def _map_browser_use_action(self, action_name: str, params: Any, element_info: Optional[Dict[str, Any]] = None) -> Optional[RecordedAction]:
        """Map a browser-use action to our RecordedAction format."""
        # Ensure params is a plain dict
        params = self._to_plain_dict(params)
        if not isinstance(params, dict):
            params = {"value": params}

        ei = element_info or {}

        def _click_params(p: Dict) -> Dict:
            result: Dict[str, Any] = {"index": p.get("index", 0)}
            if ei.get('element_text'):
                result['element_text'] = ei['element_text']
            if ei.get('element_tag'):
                result['element_tag'] = ei['element_tag']
            if ei.get('xpath'):
                result['xpath'] = ei['xpath']
            if ei.get('element_attrs'):
                result['element_attrs'] = ei['element_attrs']
            return result

        def _fill_params(p: Dict) -> Dict:
            result: Dict[str, Any] = {"index": p.get("index", 0), "value": p.get("text", "")}
            if ei.get('element_text'):
                result['element_text'] = ei['element_text']
            if ei.get('element_tag'):
                result['element_tag'] = ei['element_tag']
            if ei.get('element_attrs'):
                result['element_attrs'] = ei['element_attrs']
            return result

        def _select_params(p: Dict) -> Dict:
            result: Dict[str, Any] = {"index": p.get("index", 0), "text": p.get("text", "")}
            if ei.get('element_text'):
                result['element_text'] = ei['element_text']
            if ei.get('element_tag'):
                result['element_tag'] = ei['element_tag']
            if ei.get('element_attrs'):
                result['element_attrs'] = ei['element_attrs']
            return result

        mapping = {
            # browser-use 0.12 action names
            "navigate": ("navigate", lambda p: {"url": p.get("url", "")}),
            "click": ("click", _click_params),
            "input": ("fill", _fill_params),
            "select_dropdown": ("select", _select_params),
            "extract": ("extract", lambda p: {"query": p.get("query", "")}),
            "done": ("done", lambda p: {"text": p.get("text", ""), "success": p.get("success", False)}),
            "wait": ("wait", lambda p: {"seconds": p.get("seconds", 1)}),
            "search": ("search", lambda p: {"query": p.get("query", "")}),
            # Legacy action names
            "go_to_url": ("navigate", lambda p: {"url": p.get("url", "")}),
            "click_element": ("click", _click_params),
            "input_text": ("fill", _fill_params),
            "scroll_down": ("scroll", lambda p: {"dy": p.get("amount", 300)}),
            "scroll_up": ("scroll", lambda p: {"dy": -p.get("amount", 300)}),
            "send_keys": ("keyboard", lambda p: {"key": p.get("keys", "")}),
            "extract_content": ("extract", lambda p: {"query": "page content"}),
            "go_back": ("navigate", lambda _: {"url": "javascript:history.back()"}),
        }

        if action_name in mapping:
            step_type, param_mapper = mapping[action_name]
            try:
                mapped_params = param_mapper(params)
            except Exception:
                mapped_params = params

            # Build description with element info for clarity
            desc_parts = [action_name]
            if ei.get('element_tag'):
                desc_parts.append(ei['element_tag'])
            if ei.get('element_text'):
                desc_parts.append(f'"{ei["element_text"]}"')
            description = ' '.join(desc_parts)

            return RecordedAction(
                action_type=step_type,
                params=mapped_params,
                description=description,
            )

        # Unknown action — record as AI act
        return RecordedAction(
            action_type="ai_act",
            params={"action": action_name, **params},
            description=f"AI: {action_name}",
        )
