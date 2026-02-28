"""
AIFallback — When a deterministic step fails, use AI to recover.
Integrates with browser-use agent for single-action AI recovery,
or uses Vision AI to find elements that selectors missed.
"""

import logging
from typing import Any, Dict, Optional

from ..transport.playwright_driver import PlaywrightDriver
from ..core.task import BrowserStep
from .vision import VisionAnalyzer

logger = logging.getLogger("browgene.ai.fallback")


class AIFallback:
    """
    Handles AI-powered recovery when deterministic steps fail.

    Strategy:
    1. Take screenshot of current page state
    2. Analyze with Vision AI to understand what went wrong
    3. Either find the correct element or use browser-use for one action
    """

    def __init__(
        self,
        vision: Optional[VisionAnalyzer] = None,
        use_browser_use: bool = False,
    ):
        self.vision = vision or VisionAnalyzer()
        self.use_browser_use = use_browser_use
        self._browser_use_agent: Any = None

    async def handle_failure(
        self,
        step: BrowserStep,
        driver: PlaywrightDriver,
        resolved_params: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt to recover from a failed step.

        Args:
            step: The failed BrowserStep.
            driver: The PlaywrightDriver instance.
            resolved_params: The resolved parameters for the step.

        Returns:
            Recovery result dict, or None if recovery failed.
        """
        logger.info(f"AI FALLBACK for step: {step.description} ({step.step_type})")

        # Take screenshot for analysis
        screenshot_path = await driver.screenshot(label=f"fallback_{step.step_type}")

        # Strategy 1: Vision-based element finding (for click/fill/select failures)
        if step.step_type in ("click", "right_click", "double_click", "fill", "type", "select", "hover"):
            return await self._vision_element_recovery(step, driver, resolved_params, screenshot_path)

        # Strategy 2: Vision-based page analysis (for wait_for / assert failures)
        if step.step_type in ("wait_for", "assert"):
            return await self._vision_page_analysis(step, driver, resolved_params, screenshot_path)

        # Strategy 3: browser-use agent for complex recovery
        if self.use_browser_use:
            return await self._browser_use_recovery(step, driver, resolved_params)

        logger.warning(f"No fallback strategy for step type: {step.step_type}")
        return None

    async def _vision_element_recovery(
        self,
        step: BrowserStep,
        driver: PlaywrightDriver,
        params: Dict[str, Any],
        screenshot_path: str,
    ) -> Optional[Dict[str, Any]]:
        """Use Vision AI to find the element that the selector missed."""
        selector = params.get("selector", "")
        description = step.description or f"element matching {selector}"

        logger.info(f"Vision fallback: looking for '{description}'")

        element_info = await self.vision.find_element(screenshot_path, description)
        if not element_info or not element_info.get("found"):
            logger.warning(f"Vision could not find element: {description}")
            return None

        # Try suggested selectors first
        suggested_selectors = element_info.get("suggested_selectors", [])
        for alt_selector in suggested_selectors:
            try:
                exists = await driver.assert_element_exists(alt_selector, timeout=3000)
                if exists:
                    logger.info(f"Vision found alternative selector: {alt_selector}")
                    # Re-execute the action with the new selector
                    if step.step_type == "click":
                        await driver.click(alt_selector, button=params.get("button", "left"))
                    elif step.step_type == "right_click":
                        await driver.right_click(alt_selector)
                    elif step.step_type == "fill":
                        await driver.fill(alt_selector, params.get("value", ""))
                    elif step.step_type == "type":
                        await driver.type_text(alt_selector, params.get("text", ""))
                    elif step.step_type == "select":
                        await driver.select_option(alt_selector, value=params.get("value"))
                    elif step.step_type == "hover":
                        await driver.hover(alt_selector)

                    return {
                        "recovery_method": "vision_selector",
                        "original_selector": selector,
                        "used_selector": alt_selector,
                        "element_info": element_info,
                    }
            except Exception as e:
                logger.debug(f"Alternative selector {alt_selector} failed: {e}")
                continue

        # Fall back to coordinate click if we have coordinates
        x = element_info.get("x")
        y = element_info.get("y")
        if x is not None and y is not None:
            logger.info(f"Vision fallback: clicking coordinates ({x}, {y})")
            button = "right" if step.step_type == "right_click" else "left"
            await driver.click_coordinates(float(x), float(y), button=button)
            return {
                "recovery_method": "vision_coordinates",
                "original_selector": selector,
                "coordinates": {"x": x, "y": y},
                "element_info": element_info,
            }

        return None

    async def _vision_page_analysis(
        self,
        step: BrowserStep,
        driver: PlaywrightDriver,
        params: Dict[str, Any],
        screenshot_path: str,
    ) -> Optional[Dict[str, Any]]:
        """Analyze the page state to understand why a wait/assert failed."""
        selector = params.get("selector", "")
        prompt = f"""This browser automation step failed:
Step: {step.description}
Expected: Element matching '{selector}' to be {params.get('state', 'visible')}

Analyze the screenshot and tell me:
1. Is the page still loading?
2. Is there an error message or popup blocking the expected element?
3. What is currently visible on the page?
4. What action might resolve this issue?

Return JSON with: page_state, error_detected, suggested_action"""

        result = await self.vision.analyze_screenshot(screenshot_path, prompt)
        if result.get("success"):
            data = result.get("data", {})
            suggested_action = data.get("suggested_action", "") if isinstance(data, dict) else ""

            if suggested_action and isinstance(suggested_action, str):
                logger.info(f"Vision analysis suggests: {suggested_action}")

            return {
                "recovery_method": "vision_analysis",
                "analysis": data,
                "screenshot": screenshot_path,
            }

        return None

    async def _browser_use_recovery(
        self,
        step: BrowserStep,
        driver: PlaywrightDriver,
        params: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Use browser-use AI agent for one autonomous action."""
        try:
            from browser_use import Agent
            from langchain_openai import ChatOpenAI

            instruction = step.description or f"Perform this action: {step.step_type} with params {params}"
            logger.info(f"browser-use fallback: {instruction}")

            llm = ChatOpenAI(model="gpt-4o")
            agent = Agent(
                task=instruction,
                llm=llm,
                browser=None,  # Will need to share the browser context
            )
            result = await agent.run(max_steps=3)

            return {
                "recovery_method": "browser_use_agent",
                "instruction": instruction,
                "result": str(result),
            }
        except ImportError:
            logger.warning("browser-use not installed for AI fallback")
            return None
        except Exception as e:
            logger.error(f"browser-use fallback failed: {e}")
            return None
