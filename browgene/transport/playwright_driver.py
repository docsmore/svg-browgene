"""
PlaywrightDriver — Unified browser control layer.
Wraps Playwright async API with all the actions BrowGene needs:
click, right-click, fill, type, scroll, extract, navigate, upload, download, etc.
"""

import asyncio
import base64
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
    Download,
    FileChooser,
)

logger = logging.getLogger("browgene.transport")


class PlaywrightDriver:
    """
    Manages a Playwright browser instance and exposes all browser actions
    as simple async methods. This is the transport layer for Execute mode.
    """

    def __init__(
        self,
        headless: bool = False,
        browser_type: str = "chromium",
        slow_mo: int = 0,
        viewport_width: int = 1920,
        viewport_height: int = 1080,
        user_data_dir: Optional[str] = None,
        screenshots_dir: str = "executions/screenshots",
    ):
        self.headless = headless
        self.browser_type = browser_type
        self.slow_mo = slow_mo
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.user_data_dir = user_data_dir
        self.screenshots_dir = screenshots_dir
        os.makedirs(screenshots_dir, exist_ok=True)

        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._screenshot_counter: int = 0

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def launch(self) -> Page:
        """Launch browser and return the active page."""
        self._playwright = await async_playwright().start()

        launcher = getattr(self._playwright, self.browser_type)

        if self.user_data_dir:
            # Persistent context — reuses cookies, localStorage, etc.
            self._context = await launcher.launch_persistent_context(
                self.user_data_dir,
                headless=self.headless,
                slow_mo=self.slow_mo,
                viewport={"width": self.viewport_width, "height": self.viewport_height},
                args=["--disable-blink-features=AutomationControlled"],
            )
            self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()
        else:
            self._browser = await launcher.launch(
                headless=self.headless,
                slow_mo=self.slow_mo,
                args=["--disable-blink-features=AutomationControlled"],
            )
            self._context = await self._browser.new_context(
                viewport={"width": self.viewport_width, "height": self.viewport_height},
            )
            self._page = await self._context.new_page()

        logger.info(f"Browser launched: {self.browser_type} (headless={self.headless})")
        return self._page

    async def close(self) -> None:
        """Close browser and cleanup."""
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
            logger.info("Browser closed")
        except Exception as e:
            logger.error(f"Error closing browser: {e}")

    @property
    def page(self) -> Page:
        """Get the active page. Raises if not launched."""
        if not self._page:
            raise RuntimeError("Browser not launched. Call launch() first.")
        return self._page

    @property
    def context(self) -> BrowserContext:
        """Get the browser context."""
        if not self._context:
            raise RuntimeError("Browser not launched. Call launch() first.")
        return self._context

    # ── Navigation ─────────────────────────────────────────────────────

    async def navigate(self, url: str, wait_until: str = "domcontentloaded", timeout: int = 30000) -> None:
        """Navigate to a URL."""
        logger.info(f"NAVIGATE → {url}")
        await self.page.goto(url, wait_until=wait_until, timeout=timeout)

    async def reload(self, wait_until: str = "domcontentloaded", timeout: int = 30000) -> None:
        """Reload the current page."""
        logger.info("RELOAD page")
        await self.page.reload(wait_until=wait_until, timeout=timeout)

    async def go_back(self) -> None:
        """Navigate back."""
        await self.page.go_back()

    async def go_forward(self) -> None:
        """Navigate forward."""
        await self.page.go_forward()

    def current_url(self) -> str:
        """Get the current page URL."""
        return self.page.url

    # ── Click Actions ──────────────────────────────────────────────────

    async def click(
        self,
        selector: str,
        button: str = "left",
        click_count: int = 1,
        timeout: int = 30000,
        force: bool = False,
        position: Optional[Dict[str, float]] = None,
    ) -> None:
        """Click an element. Supports left, right, middle buttons."""
        logger.info(f"CLICK ({button}) → {selector}")
        kwargs: Dict[str, Any] = {
            "button": button,
            "click_count": click_count,
            "timeout": timeout,
            "force": force,
        }
        if position:
            kwargs["position"] = position
        await self.page.click(selector, **kwargs)

    async def right_click(self, selector: str, timeout: int = 30000) -> None:
        """Right-click an element to open context menu."""
        logger.info(f"RIGHT_CLICK → {selector}")
        await self.page.click(selector, button="right", timeout=timeout)

    async def double_click(self, selector: str, timeout: int = 30000) -> None:
        """Double-click an element."""
        logger.info(f"DOUBLE_CLICK → {selector}")
        await self.page.dblclick(selector, timeout=timeout)

    async def hover(self, selector: str, timeout: int = 30000) -> None:
        """Hover over an element."""
        logger.info(f"HOVER → {selector}")
        await self.page.hover(selector, timeout=timeout)

    async def click_coordinates(
        self, x: float, y: float, button: str = "left", click_count: int = 1
    ) -> None:
        """Click at specific page coordinates (for when selectors won't work)."""
        logger.info(f"CLICK_COORDS ({button}) → ({x}, {y})")
        await self.page.mouse.click(x, y, button=button, click_count=click_count)

    # ── Input Actions ──────────────────────────────────────────────────

    async def fill(self, selector: str, value: str, timeout: int = 30000) -> None:
        """Fill an input field (clears existing value first)."""
        logger.info(f"FILL → {selector} = '{value[:50]}...' " if len(value) > 50 else f"FILL → {selector} = '{value}'")
        await self.page.fill(selector, value, timeout=timeout)

    async def type_text(
        self, selector: str, text: str, delay: int = 50, timeout: int = 30000
    ) -> None:
        """Type text character by character (simulates real typing)."""
        logger.info(f"TYPE → {selector} = '{text[:50]}...' " if len(text) > 50 else f"TYPE → {selector} = '{text}'")
        await self.page.click(selector, timeout=timeout)
        await self.page.type(selector, text, delay=delay)

    async def clear(self, selector: str, timeout: int = 30000) -> None:
        """Clear an input field."""
        logger.info(f"CLEAR → {selector}")
        await self.page.fill(selector, "", timeout=timeout)

    async def select_option(
        self, selector: str, value: Optional[str] = None, label: Optional[str] = None,
        index: Optional[int] = None, timeout: int = 30000
    ) -> List[str]:
        """Select an option from a dropdown."""
        logger.info(f"SELECT → {selector} (value={value}, label={label}, index={index})")
        kwargs: Dict[str, Any] = {"timeout": timeout}
        if value is not None:
            kwargs["value"] = value
        elif label is not None:
            kwargs["label"] = label
        elif index is not None:
            kwargs["index"] = index
        return await self.page.select_option(selector, **kwargs)

    async def check(self, selector: str, timeout: int = 30000) -> None:
        """Check a checkbox."""
        logger.info(f"CHECK → {selector}")
        await self.page.check(selector, timeout=timeout)

    async def uncheck(self, selector: str, timeout: int = 30000) -> None:
        """Uncheck a checkbox."""
        logger.info(f"UNCHECK → {selector}")
        await self.page.uncheck(selector, timeout=timeout)

    # ── Keyboard Actions ───────────────────────────────────────────────

    async def press_key(self, key: str, selector: Optional[str] = None) -> None:
        """Press a keyboard key (Tab, Enter, Escape, ArrowDown, etc.)."""
        logger.info(f"KEYBOARD → {key}" + (f" on {selector}" if selector else ""))
        if selector:
            await self.page.press(selector, key)
        else:
            await self.page.keyboard.press(key)

    async def key_down(self, key: str) -> None:
        """Hold down a key."""
        await self.page.keyboard.down(key)

    async def key_up(self, key: str) -> None:
        """Release a key."""
        await self.page.keyboard.up(key)

    async def keyboard_type(self, text: str, delay: int = 50) -> None:
        """Type text via keyboard (no selector needed, types wherever focus is)."""
        logger.info(f"KEYBOARD_TYPE → '{text[:50]}...' " if len(text) > 50 else f"KEYBOARD_TYPE → '{text}'")
        await self.page.keyboard.type(text, delay=delay)

    # ── Scroll Actions ─────────────────────────────────────────────────

    async def scroll(
        self, dx: int = 0, dy: int = 0, selector: Optional[str] = None
    ) -> None:
        """Scroll the page or a specific element."""
        if selector:
            logger.info(f"SCROLL → {selector} ({dx}, {dy})")
            await self.page.evaluate(
                f"document.querySelector('{selector}').scrollBy({dx}, {dy})"
            )
        else:
            logger.info(f"SCROLL page ({dx}, {dy})")
            await self.page.mouse.wheel(dx, dy)

    async def scroll_to_bottom(self) -> None:
        """Scroll to the bottom of the page."""
        await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

    async def scroll_to_top(self) -> None:
        """Scroll to the top of the page."""
        await self.page.evaluate("window.scrollTo(0, 0)")

    async def scroll_into_view(self, selector: str) -> None:
        """Scroll an element into view."""
        await self.page.evaluate(
            f"document.querySelector('{selector}')?.scrollIntoView({{behavior: 'smooth', block: 'center'}})"
        )

    # ── Wait Actions ───────────────────────────────────────────────────

    async def wait_for_selector(
        self, selector: str, state: str = "visible", timeout: int = 30000
    ) -> None:
        """Wait for an element to reach a state (visible, hidden, attached, detached)."""
        logger.info(f"WAIT_FOR → {selector} (state={state}, timeout={timeout}ms)")
        await self.page.wait_for_selector(selector, state=state, timeout=timeout)

    async def wait_for_navigation(self, wait_until: str = "domcontentloaded", timeout: int = 30000) -> None:
        """Wait for navigation to complete."""
        await self.page.wait_for_load_state(wait_until, timeout=timeout)

    async def wait(self, seconds: float) -> None:
        """Fixed delay."""
        logger.info(f"WAIT → {seconds}s")
        await asyncio.sleep(seconds)

    async def wait_for_url(self, url_pattern: str, timeout: int = 30000) -> None:
        """Wait for the URL to match a pattern."""
        await self.page.wait_for_url(url_pattern, timeout=timeout)

    # ── Extract / Evaluate ─────────────────────────────────────────────

    async def extract(self, js_expression: str) -> Any:
        """Evaluate JavaScript and return the result."""
        logger.info(f"EXTRACT → {js_expression[:80]}...")
        result = await self.page.evaluate(js_expression)
        return result

    async def extract_text(self, selector: str) -> str:
        """Get the text content of an element."""
        return await self.page.text_content(selector) or ""

    async def extract_attribute(self, selector: str, attribute: str) -> Optional[str]:
        """Get an attribute value of an element."""
        return await self.page.get_attribute(selector, attribute)

    async def extract_inner_html(self, selector: str) -> str:
        """Get the inner HTML of an element."""
        return await self.page.inner_html(selector)

    async def extract_all_text(self, selector: str) -> List[str]:
        """Get text content from all matching elements."""
        elements = await self.page.query_selector_all(selector)
        texts = []
        for el in elements:
            text = await el.text_content()
            if text:
                texts.append(text.strip())
        return texts

    async def evaluate(self, js_expression: str) -> Any:
        """Run arbitrary JavaScript on the page."""
        logger.info(f"EVALUATE → {js_expression[:80]}...")
        return await self.page.evaluate(js_expression)

    async def query_selector_count(self, selector: str) -> int:
        """Count elements matching a selector."""
        elements = await self.page.query_selector_all(selector)
        return len(elements)

    async def is_visible(self, selector: str) -> bool:
        """Check if an element is visible."""
        return await self.page.is_visible(selector)

    async def is_checked(self, selector: str) -> bool:
        """Check if a checkbox/radio is checked."""
        return await self.page.is_checked(selector)

    # ── File Upload / Download ─────────────────────────────────────────

    async def upload_file(self, selector: str, file_paths: List[str], timeout: int = 30000) -> None:
        """Upload files to a file input."""
        logger.info(f"UPLOAD → {selector} files={file_paths}")
        await self.page.set_input_files(selector, file_paths, timeout=timeout)

    async def download(self, trigger_selector: str, save_path: str, timeout: int = 30000) -> str:
        """Click a download trigger and save the file."""
        logger.info(f"DOWNLOAD → clicking {trigger_selector}")
        async with self.page.expect_download(timeout=timeout) as download_info:
            await self.page.click(trigger_selector)
        download: Download = await download_info.value
        await download.save_as(save_path)
        logger.info(f"DOWNLOAD saved → {save_path}")
        return save_path

    # ── Frame / Tab Management ─────────────────────────────────────────

    async def switch_to_frame(self, selector: str) -> None:
        """Switch context to an iframe."""
        logger.info(f"FRAME_SWITCH → {selector}")
        frame = self.page.frame_locator(selector)
        # Store reference for subsequent actions
        self._active_frame = frame

    async def switch_to_main_frame(self) -> None:
        """Switch back to main frame."""
        self._active_frame = None

    async def open_new_tab(self, url: Optional[str] = None) -> Page:
        """Open a new tab and switch to it."""
        page = await self.context.new_page()
        if url:
            await page.goto(url)
        self._page = page
        logger.info(f"TAB_OPEN → new tab" + (f" ({url})" if url else ""))
        return page

    async def switch_tab(self, index: int) -> Page:
        """Switch to a specific tab by index."""
        pages = self.context.pages
        if 0 <= index < len(pages):
            self._page = pages[index]
            await self._page.bring_to_front()
            logger.info(f"TAB_SWITCH → tab {index}")
            return self._page
        raise IndexError(f"Tab index {index} out of range (have {len(pages)} tabs)")

    async def close_tab(self, index: Optional[int] = None) -> None:
        """Close a tab by index, or current tab if no index."""
        if index is not None:
            pages = self.context.pages
            if 0 <= index < len(pages):
                await pages[index].close()
                logger.info(f"TAB_CLOSE → tab {index}")
        else:
            await self.page.close()
            logger.info("TAB_CLOSE → current tab")
        # Switch to remaining tab if available
        pages = self.context.pages
        if pages:
            self._page = pages[-1]

    async def get_tab_count(self) -> int:
        """Get the number of open tabs."""
        return len(self.context.pages)

    # ── Drag & Drop ────────────────────────────────────────────────────

    async def drag_drop(
        self, source_selector: str, target_selector: str, timeout: int = 30000
    ) -> None:
        """Drag an element and drop it on another."""
        logger.info(f"DRAG_DROP → {source_selector} → {target_selector}")
        await self.page.drag_and_drop(source_selector, target_selector, timeout=timeout)

    # ── Screenshots ────────────────────────────────────────────────────

    async def screenshot(
        self,
        label: str = "screenshot",
        full_page: bool = False,
        selector: Optional[str] = None,
    ) -> str:
        """Take a screenshot and save it. Returns the file path."""
        self._screenshot_counter += 1
        filename = f"{self._screenshot_counter:04d}_{label}.png"
        filepath = os.path.join(self.screenshots_dir, filename)

        if selector:
            element = await self.page.query_selector(selector)
            if element:
                await element.screenshot(path=filepath)
            else:
                await self.page.screenshot(path=filepath, full_page=full_page)
        else:
            await self.page.screenshot(path=filepath, full_page=full_page)

        logger.info(f"SCREENSHOT → {filepath}")
        return filepath

    async def screenshot_base64(self, full_page: bool = False) -> str:
        """Take a screenshot and return as base64 string."""
        screenshot_bytes = await self.page.screenshot(full_page=full_page)
        return base64.b64encode(screenshot_bytes).decode("utf-8")

    # ── Assert / Condition Checks ──────────────────────────────────────

    async def assert_element_exists(self, selector: str, timeout: int = 5000) -> bool:
        """Check if an element exists on the page."""
        try:
            await self.page.wait_for_selector(selector, timeout=timeout)
            return True
        except Exception:
            return False

    async def assert_text_contains(self, selector: str, text: str) -> bool:
        """Check if an element contains specific text."""
        content = await self.page.text_content(selector)
        return text in (content or "")

    async def assert_url_contains(self, text: str) -> bool:
        """Check if the current URL contains specific text."""
        return text in self.page.url

    # ── Page Info ──────────────────────────────────────────────────────

    async def get_page_title(self) -> str:
        """Get the current page title."""
        return await self.page.title()

    async def get_page_content(self) -> str:
        """Get the full HTML content of the page."""
        return await self.page.content()

    async def get_cookies(self) -> List[Dict[str, Any]]:
        """Get all cookies for the current context."""
        return await self.context.cookies()

    async def set_cookies(self, cookies: List[Dict[str, Any]]) -> None:
        """Set cookies on the current context."""
        await self.context.add_cookies(cookies)
