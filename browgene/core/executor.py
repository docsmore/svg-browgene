"""
Executor — Execute mode step runner.
Takes a BrowserTask and runs each BrowserStep via PlaywrightDriver.
Handles parameter interpolation, retries, AI fallback, and memory.
"""

import asyncio
import json
import logging
import re
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

from ..memory.execution_memory import ExecutionMemory
from ..transport.playwright_driver import PlaywrightDriver
from .task import BrowserStep, BrowserTask, OnFailure, StepStatus, StepType, TaskExecution

logger = logging.getLogger("browgene.executor")

# Pattern for ${variable_name} interpolation
PARAM_PATTERN = re.compile(r"\$\{(\w+(?:\.\w+)*)\}")


def _resolve_value(value: Any, parameters: Dict[str, Any], memory: ExecutionMemory) -> Any:
    """Resolve ${variable} references in a value against parameters and memory."""
    if isinstance(value, str):
        def replacer(match: re.Match) -> str:
            var_path = match.group(1)
            parts = var_path.split(".")
            # First check parameters, then memory
            if parts[0] in parameters:
                result = parameters[parts[0]]
                for part in parts[1:]:
                    if isinstance(result, dict):
                        result = result.get(part, match.group(0))
                    else:
                        return match.group(0)
                return str(result)
            if memory.has(parts[0]):
                result = memory.get(parts[0])
                for part in parts[1:]:
                    if isinstance(result, dict):
                        result = result.get(part, match.group(0))
                    else:
                        return match.group(0)
                return str(result)
            return match.group(0)
        return PARAM_PATTERN.sub(replacer, value)
    elif isinstance(value, dict):
        return {k: _resolve_value(v, parameters, memory) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_value(item, parameters, memory) for item in value]
    return value


class Executor:
    """
    Runs a BrowserTask step-by-step using PlaywrightDriver.
    Supports parameter interpolation, retries, AI fallback, and memory.
    """

    def __init__(
        self,
        driver: PlaywrightDriver,
        ai_fallback_fn: Optional[Callable] = None,
        on_step_start: Optional[Callable] = None,
        on_step_end: Optional[Callable] = None,
    ):
        self.driver = driver
        self.ai_fallback_fn = ai_fallback_fn
        self.on_step_start = on_step_start
        self.on_step_end = on_step_end

    async def execute_task(
        self,
        task: BrowserTask,
        parameters: Optional[Dict[str, Any]] = None,
        from_step: int = 0,
    ) -> TaskExecution:
        """
        Execute a full BrowserTask.

        Args:
            task: The task to execute.
            parameters: Runtime parameter values to resolve ${variables}.
            from_step: Step index to resume from (for checkpoint/resume).

        Returns:
            TaskExecution with results, memory, screenshots, and extracted data.
        """
        params = parameters or {}
        memory = ExecutionMemory()
        execution = TaskExecution(
            task_name=task.name,
            parameters=params,
            mode="execute",
            status="running",
            steps_total=len(task.steps),
            start_time=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        logger.info(f"═══ EXECUTING TASK: {task.name} ({len(task.steps)} steps, from_step={from_step}) ═══")

        # Navigate to start URL if defined
        if task.start_url and from_step == 0:
            resolved_url = _resolve_value(task.start_url, params, memory)
            await self.driver.navigate(resolved_url)

        for i in range(from_step, len(task.steps)):
            step = task.steps[i]
            step_start = time.time()

            # Notify step start
            if self.on_step_start:
                await self._maybe_await(self.on_step_start, i, step, execution)

            # Resolve parameters in step params
            resolved_params = _resolve_value(step.params, params, memory)

            logger.info(f"── Step {i + 1}/{len(task.steps)}: {step}")

            # Take screenshot before step
            screenshot_before = None
            if step.take_screenshot:
                try:
                    screenshot_before = await self.driver.screenshot(
                        label=f"step{i + 1:03d}_before_{step.step_type}"
                    )
                    step.screenshot_before = screenshot_before
                except Exception as e:
                    logger.warning(f"Failed to take before-screenshot: {e}")

            # Execute with retries
            success = False
            error_msg = None
            retry_count = 0

            while retry_count <= step.max_retries and not success:
                if retry_count > 0:
                    logger.info(f"   Retry {retry_count}/{step.max_retries}")
                    await asyncio.sleep(0.5)

                try:
                    await self._execute_step(step.step_type, resolved_params, memory, step.timeout_ms)
                    success = True
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"   Step failed: {error_msg}")
                    retry_count += 1

            # Handle failure
            if not success:
                on_fail = step.on_failure

                if on_fail == OnFailure.AI_FALLBACK.value and self.ai_fallback_fn:
                    logger.info(f"   → AI FALLBACK for: {step.description}")
                    try:
                        ai_result = await self._maybe_await(
                            self.ai_fallback_fn, step, self.driver, resolved_params
                        )
                        if ai_result:
                            success = True
                            step.status = StepStatus.AI_FALLBACK.value
                            step.ai_analysis_result = ai_result if isinstance(ai_result, dict) else {"result": ai_result}
                    except Exception as ai_err:
                        logger.error(f"   AI fallback also failed: {ai_err}")

                elif on_fail == OnFailure.SKIP.value:
                    logger.info(f"   → SKIPPING step (on_failure=skip)")
                    step.status = StepStatus.SKIPPED.value
                    success = True  # Continue to next step

                elif on_fail == OnFailure.STOP.value:
                    step.status = StepStatus.FAILED.value
                    step.error = error_msg
                    step.duration_ms = int((time.time() - step_start) * 1000)
                    execution.status = "failed"
                    execution.error = f"Step {i + 1} failed: {error_msg}"
                    execution.steps_completed = i
                    execution.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    execution.memory = memory.to_dict()
                    execution.step_results.append(self._step_result(step, i))
                    logger.error(f"═══ TASK FAILED at step {i + 1}: {error_msg} ═══")

                    if self.on_step_end:
                        await self._maybe_await(self.on_step_end, i, step, execution)
                    return execution

            # Mark success
            if success and step.status == StepStatus.PENDING.value:
                step.status = StepStatus.SUCCESS.value

            step.duration_ms = int((time.time() - step_start) * 1000)

            # Take screenshot after step
            if step.take_screenshot:
                try:
                    screenshot_after = await self.driver.screenshot(
                        label=f"step{i + 1:03d}_after_{step.step_type}"
                    )
                    step.screenshot_after = screenshot_after
                    execution.screenshots.append(screenshot_after)
                except Exception as e:
                    logger.warning(f"Failed to take after-screenshot: {e}")

            # Collect extracted data
            if step.extracted_data is not None:
                execution.extracted_data.append({
                    "step": i + 1,
                    "data": step.extracted_data,
                })

            execution.steps_completed = i + 1
            execution.step_results.append(self._step_result(step, i))

            # Notify step end
            if self.on_step_end:
                await self._maybe_await(self.on_step_end, i, step, execution)

            # Post-step delay
            if step.delay_ms > 0:
                await asyncio.sleep(step.delay_ms / 1000.0)

        # All steps completed
        execution.status = "completed"
        execution.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        execution.memory = memory.to_dict()
        logger.info(f"═══ TASK COMPLETED: {task.name} ({execution.steps_completed} steps) ═══")
        return execution

    async def _execute_step(
        self,
        step_type: str,
        params: Dict[str, Any],
        memory: ExecutionMemory,
        timeout_ms: int,
    ) -> None:
        """Execute a single step type against the PlaywrightDriver."""
        d = self.driver

        if step_type == StepType.NAVIGATE.value:
            await d.navigate(
                url=params["url"],
                wait_until=params.get("wait_until", "domcontentloaded"),
                timeout=params.get("timeout", timeout_ms),
            )

        elif step_type == StepType.CLICK.value:
            if "selector" in params:
                kwargs: Dict[str, Any] = {"selector": params["selector"], "timeout": timeout_ms}
                if "button" in params:
                    kwargs["button"] = params["button"]
                if "position" in params:
                    kwargs["position"] = params["position"]
                if "force" in params:
                    kwargs["force"] = params["force"]
                await d.click(**kwargs)
            elif "element_text" in params:
                selector = self._build_text_selector(params)
                logger.info(f"CLICK (text) → {selector}")
                await d.page.click(selector, timeout=timeout_ms)
            elif "index" in params:
                await self._click_by_index(d, params["index"], timeout_ms)

        elif step_type == StepType.RIGHT_CLICK.value:
            await d.right_click(params["selector"], timeout=timeout_ms)

        elif step_type == StepType.DOUBLE_CLICK.value:
            await d.double_click(params["selector"], timeout=timeout_ms)

        elif step_type == StepType.HOVER.value:
            await d.hover(params["selector"], timeout=timeout_ms)

        elif step_type == StepType.FILL.value:
            if "selector" in params:
                await d.fill(params["selector"], params["value"], timeout=timeout_ms)
            elif "element_text" in params or "element_attrs" in params:
                selector = self._build_input_selector(params)
                logger.info(f"FILL (text) → {selector} = '{params.get('value', '')[:40]}'")
                await d.page.fill(selector, params.get("value", ""), timeout=timeout_ms)
            elif "index" in params:
                await self._fill_by_index(d, params["index"], params.get("value", ""), timeout_ms)

        elif step_type == StepType.TYPE.value:
            await d.type_text(
                params["selector"],
                params["text"],
                delay=params.get("delay", 50),
                timeout=timeout_ms,
            )

        elif step_type == StepType.SELECT.value:
            if "selector" in params:
                await d.select_option(
                    params["selector"],
                    value=params.get("value"),
                    label=params.get("label"),
                    index=params.get("index"),
                    timeout=timeout_ms,
                )
            elif "element_attrs" in params or "element_text" in params:
                selector = self._build_input_selector(params)
                logger.info(f"SELECT (text) → {selector} label='{params.get('text', '')}'")
                await d.page.select_option(selector, label=params.get("text", ""), timeout=timeout_ms)
            elif "index" in params and "text" in params:
                await self._select_by_index(d, params["index"], params["text"], timeout_ms)

        elif step_type == StepType.SCROLL.value:
            await d.scroll(
                dx=params.get("dx", 0),
                dy=params.get("dy", 0),
                selector=params.get("selector"),
            )

        elif step_type == StepType.WAIT_FOR.value:
            await d.wait_for_selector(
                params["selector"],
                state=params.get("state", "visible"),
                timeout=params.get("timeout", timeout_ms),
            )

        elif step_type == StepType.WAIT.value:
            seconds = params.get("seconds", 1.0)
            if "time_ms" in params:
                seconds = params["time_ms"] / 1000.0
            await d.wait(seconds)

        elif step_type == StepType.SCREENSHOT.value:
            await d.screenshot(
                label=params.get("label", "manual"),
                full_page=params.get("full_page", False),
                selector=params.get("selector"),
            )

        elif step_type == StepType.EXTRACT.value:
            if "js_expression" in params:
                result = await d.extract(params["js_expression"])
            else:
                # browser-use 0.12 style: extract with a query — get page text
                result = await d.extract("document.body.innerText")
            # Parse JSON string results
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except (json.JSONDecodeError, ValueError):
                    pass
            # Store extracted data on the step (will be collected by executor)
            # Access the step via closure in execute_task
            return result

        elif step_type == StepType.KEYBOARD.value:
            key = params.get("key", "")
            selector = params.get("selector")
            await d.press_key(key, selector=selector)

        elif step_type == StepType.UPLOAD.value:
            await d.upload_file(
                params["selector"],
                params["file_paths"],
                timeout=timeout_ms,
            )

        elif step_type == StepType.DOWNLOAD.value:
            path = await d.download(
                params["trigger_selector"],
                params["save_path"],
                timeout=timeout_ms,
            )
            return path

        elif step_type == StepType.EVALUATE.value:
            result = await d.evaluate(params["js_expression"])
            return result

        elif step_type == StepType.HOLD_DATA.value:
            key = params["key"]
            if "value" in params:
                memory.hold(key, params["value"])
            elif "extract_js" in params:
                value = await d.extract(params["extract_js"])
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        pass
                memory.hold(key, value)
            elif "from_step" in params:
                # Reference extracted data from a previous step (handled at task level)
                logger.info(f"HOLD_DATA from step reference — handled at task level")

        elif step_type == StepType.MERGE_DATA.value:
            memory.merge(
                sources=params["sources"],
                target=params["target"],
                strategy=params.get("strategy", "combine"),
            )

        elif step_type == StepType.ASSERT.value:
            condition = params.get("condition", "element_exists")
            if condition == "element_exists":
                exists = await d.assert_element_exists(params["selector"], timeout=5000)
                if not exists:
                    raise AssertionError(f"Element not found: {params['selector']}")
            elif condition == "text_contains":
                contains = await d.assert_text_contains(params["selector"], params["text"])
                if not contains:
                    raise AssertionError(f"Text '{params['text']}' not found in {params['selector']}")
            elif condition == "url_contains":
                contains = await d.assert_url_contains(params["text"])
                if not contains:
                    raise AssertionError(f"URL does not contain '{params['text']}'")

        elif step_type == StepType.FRAME_SWITCH.value:
            if params.get("main", False):
                await d.switch_to_main_frame()
            else:
                await d.switch_to_frame(params["selector"])

        elif step_type == StepType.TAB_OPEN.value:
            await d.open_new_tab(url=params.get("url"))

        elif step_type == StepType.TAB_SWITCH.value:
            await d.switch_tab(params["index"])

        elif step_type == StepType.TAB_CLOSE.value:
            await d.close_tab(index=params.get("index"))

        elif step_type == StepType.DRAG_DROP.value:
            await d.drag_drop(
                params["source_selector"],
                params["target_selector"],
                timeout=timeout_ms,
            )

        elif step_type == StepType.AI_ANALYZE.value:
            # Delegate to AI vision — caller should provide ai_fallback_fn
            logger.info(f"AI_ANALYZE step — requires AI integration")
            # Take screenshot for analysis
            screenshot_path = await d.screenshot(label="ai_analyze")
            return {"screenshot": screenshot_path, "prompt": params.get("prompt", "")}

        elif step_type == StepType.AI_ACT.value:
            # Delegate to browser-use agent for one action
            logger.info(f"AI_ACT step — requires browser-use integration")
            return {"instruction": params.get("instruction", "")}

        elif step_type == StepType.DONE.value:
            # Completion marker from browser-use exploration — no action needed
            logger.info(f"DONE step: {params.get('text', 'Task completed')[:100]}")

        elif step_type == StepType.SEARCH.value:
            # Search action — treat as navigation to search URL or log
            query = params.get("query", "")
            if query:
                logger.info(f"SEARCH: {query}")

        else:
            raise ValueError(f"Unknown step type: {step_type}")

    def _step_result(self, step: BrowserStep, index: int) -> Dict[str, Any]:
        """Create a step result summary."""
        return {
            "step_index": index,
            "step_type": step.step_type,
            "description": step.description,
            "status": step.status,
            "duration_ms": step.duration_ms,
            "error": step.error,
            "has_screenshot_before": step.screenshot_before is not None,
            "has_screenshot_after": step.screenshot_after is not None,
            "has_extracted_data": step.extracted_data is not None,
        }

    @staticmethod
    def _build_text_selector(params: Dict[str, Any]) -> str:
        """Build a Playwright text-based selector from element info."""
        tag = params.get('element_tag', '')
        text = params.get('element_text', '')
        attrs = params.get('element_attrs', {})

        # Try id first (most stable)
        if attrs.get('id'):
            return f'#{attrs["id"]}'

        # Try data-testid
        if attrs.get('data-testid'):
            return f'[data-testid="{attrs["data-testid"]}"]'

        # Use role-based selector if we know the tag
        if tag in ('button', 'a', 'link') and text:
            role = 'link' if tag == 'a' else 'button'
            return f'role={role}[name="{text}"]'

        # Generic text match
        if text:
            if tag:
                return f'{tag}:has-text("{text}")'
            return f'text="{text}"'

        # Fallback to tag
        return tag or '*'

    @staticmethod
    def _build_input_selector(params: Dict[str, Any]) -> str:
        """Build a Playwright selector for input/select elements."""
        attrs = params.get('element_attrs', {})
        tag = params.get('element_tag', 'input')

        if attrs.get('id'):
            return f'#{attrs["id"]}'
        if attrs.get('name'):
            return f'{tag}[name="{attrs["name"]}"]'
        if attrs.get('placeholder'):
            return f'{tag}[placeholder="{attrs["placeholder"]}"]'
        if attrs.get('aria-label'):
            return f'{tag}[aria-label="{attrs["aria-label"]}"]'
        if attrs.get('type'):
            return f'{tag}[type="{attrs["type"]}"]'

        # Fall back to element_text (label)
        text = params.get('element_text', '')
        if text:
            return f'{tag}:has-text("{text}")'

        return tag

    async def _click_by_index(self, d: Any, index: int, timeout_ms: int) -> None:
        """Click an interactive element by its browser-use style index."""
        js = """
        (idx) => {
            const els = document.querySelectorAll(
                'a, button, input, select, textarea, [role="button"], [onclick], [tabindex]'
            );
            const visible = Array.from(els).filter(el => {
                const r = el.getBoundingClientRect();
                return r.width > 0 && r.height > 0 && el.offsetParent !== null;
            });
            if (idx < visible.length) {
                visible[idx].click();
                return true;
            }
            return false;
        }
        """
        clicked = await d.page.evaluate(js, index)
        if not clicked:
            logger.warning(f"Index {index} not found among interactive elements, trying all visible")
            js_all = """
            (idx) => {
                const all = document.querySelectorAll('*');
                const visible = Array.from(all).filter(el => {
                    const r = el.getBoundingClientRect();
                    return r.width > 0 && r.height > 0 && typeof el.click === 'function';
                });
                if (idx < visible.length) {
                    visible[idx].click();
                    return true;
                }
                return false;
            }
            """
            await d.page.evaluate(js_all, index)
        await asyncio.sleep(0.3)

    async def _fill_by_index(self, d: Any, index: int, value: str, timeout_ms: int) -> None:
        """Fill an input element by its browser-use style index."""
        js = """
        (args) => {
            const [idx, val] = args;
            const els = document.querySelectorAll('input, textarea, select, [contenteditable="true"]');
            const visible = Array.from(els).filter(el => {
                const r = el.getBoundingClientRect();
                return r.width > 0 && r.height > 0 && el.offsetParent !== null;
            });
            if (idx < visible.length) {
                const el = visible[idx];
                el.focus();
                el.value = val;
                el.dispatchEvent(new Event('input', {bubbles: true}));
                el.dispatchEvent(new Event('change', {bubbles: true}));
                return true;
            }
            return false;
        }
        """
        filled = await d.page.evaluate(js, [index, value])
        if not filled:
            logger.warning(f"Input index {index} not found")
        await asyncio.sleep(0.3)

    async def _select_by_index(self, d: Any, index: int, text: str, timeout_ms: int) -> None:
        """Select a dropdown option by element index and option text."""
        js = """
        (args) => {
            const [idx, text] = args;
            const selects = document.querySelectorAll('select');
            const visible = Array.from(selects).filter(el => {
                const r = el.getBoundingClientRect();
                return r.width > 0 && r.height > 0;
            });
            // Try by absolute index first, then by select-only index
            let target = null;
            const allEls = document.querySelectorAll(
                'a, button, input, select, textarea, [role="button"], [onclick], [tabindex]'
            );
            const visibleAll = Array.from(allEls).filter(el => {
                const r = el.getBoundingClientRect();
                return r.width > 0 && r.height > 0 && el.offsetParent !== null;
            });
            if (idx < visibleAll.length && visibleAll[idx].tagName === 'SELECT') {
                target = visibleAll[idx];
            }
            if (!target && visible.length > 0) {
                target = visible[0];
            }
            if (target) {
                const options = Array.from(target.options);
                const opt = options.find(o => o.text.trim() === text || o.value === text);
                if (opt) {
                    target.value = opt.value;
                    target.dispatchEvent(new Event('change', {bubbles: true}));
                    return true;
                }
            }
            return false;
        }
        """
        selected = await d.page.evaluate(js, [index, text])
        if not selected:
            logger.warning(f"Select index {index} with text '{text}' not found")
        await asyncio.sleep(0.5)

    async def _maybe_await(self, fn: Callable, *args: Any) -> Any:
        """Call a function, awaiting it if it's a coroutine."""
        result = fn(*args)
        if asyncio.iscoroutine(result):
            return await result
        return result
