"""
VisionAnalyzer — AI-powered screenshot analysis.
Uses Gemini or GPT-4V to analyze page screenshots for element detection,
data extraction, and page understanding.
"""

import base64
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("browgene.ai.vision")


class VisionAnalyzer:
    """
    Analyzes browser screenshots using vision LLMs (Gemini, GPT-4V).
    Used for AI fallback when deterministic steps fail,
    and for AI_ANALYZE step type.
    """

    def __init__(self, provider: str = "gemini", model: Optional[str] = None, api_key: Optional[str] = None):
        self.provider = provider
        self.model = model
        self.api_key = api_key or self._get_api_key()
        self._client: Any = None

    def _get_api_key(self) -> str:
        """Get API key from environment."""
        if self.provider == "gemini":
            return os.getenv("GOOGLE_API_KEY", "")
        elif self.provider == "openai":
            return os.getenv("OPENAI_API_KEY", "")
        return ""

    def _get_client(self) -> Any:
        """Lazily initialize the AI client."""
        if self._client:
            return self._client

        if self.provider == "gemini":
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model or "gemini-2.0-flash")
                logger.info(f"Gemini Vision client initialized: {self.model or 'gemini-2.0-flash'}")
            except ImportError:
                logger.error("google-generativeai not installed. Run: pip install google-generativeai")
                raise
        elif self.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
                logger.info(f"OpenAI Vision client initialized: {self.model or 'gpt-4o'}")
            except ImportError:
                logger.error("openai not installed. Run: pip install openai")
                raise

        return self._client

    def _image_to_base64(self, image_path: str) -> str:
        """Convert an image file to base64 string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    async def analyze_screenshot(
        self,
        screenshot_path: str,
        prompt: str,
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a screenshot with a custom prompt.

        Args:
            screenshot_path: Path to the screenshot image.
            prompt: What to analyze in the screenshot.
            output_schema: Optional JSON schema for structured output.

        Returns:
            Dict with 'success', 'data', and optional 'error'.
        """
        try:
            client = self._get_client()
            full_prompt = prompt
            if output_schema:
                full_prompt += f"\n\nReturn your response as JSON matching this schema:\n{json.dumps(output_schema, indent=2)}"

            if self.provider == "gemini":
                return await self._analyze_gemini(client, screenshot_path, full_prompt)
            elif self.provider == "openai":
                return await self._analyze_openai(client, screenshot_path, full_prompt)
            else:
                return {"success": False, "error": f"Unknown provider: {self.provider}"}

        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return {"success": False, "error": str(e)}

    async def _analyze_gemini(self, client: Any, screenshot_path: str, prompt: str) -> Dict[str, Any]:
        """Analyze with Gemini Vision."""
        import PIL.Image
        image = PIL.Image.open(screenshot_path)
        response = client.generate_content([prompt, image])
        text = response.text

        # Try to parse as JSON
        data = self._try_parse_json(text)
        return {"success": True, "data": data, "raw_text": text}

    async def _analyze_openai(self, client: Any, screenshot_path: str, prompt: str) -> Dict[str, Any]:
        """Analyze with OpenAI GPT-4V."""
        b64_image = self._image_to_base64(screenshot_path)
        response = client.chat.completions.create(
            model=self.model or "gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=4096,
        )
        text = response.choices[0].message.content or ""
        data = self._try_parse_json(text)
        return {"success": True, "data": data, "raw_text": text}

    async def find_element(
        self,
        screenshot_path: str,
        element_description: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Find a UI element in a screenshot by description.
        Returns coordinates and selector suggestions.
        """
        prompt = f"""Analyze this screenshot and find the UI element described as: "{element_description}"

Return a JSON object with:
- "found": true/false
- "x": approximate x coordinate (pixels from left)
- "y": approximate y coordinate (pixels from top)
- "description": what you see at that location
- "suggested_selectors": list of CSS selectors that might target this element
- "confidence": 0.0 to 1.0"""

        result = await self.analyze_screenshot(screenshot_path, prompt)
        if result.get("success") and isinstance(result.get("data"), dict):
            return result["data"]
        return None

    async def extract_page_data(
        self,
        screenshot_path: str,
        extraction_prompt: str,
    ) -> Dict[str, Any]:
        """Extract structured data from a page screenshot."""
        prompt = f"""Analyze this screenshot and extract the following data:
{extraction_prompt}

Return the extracted data as a JSON object."""

        return await self.analyze_screenshot(screenshot_path, prompt)

    def _try_parse_json(self, text: str) -> Any:
        """Try to extract and parse JSON from text."""
        # Try direct parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try extracting from markdown code block
        import re
        json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        # Return raw text if can't parse
        return text
