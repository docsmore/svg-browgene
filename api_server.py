#!/usr/bin/env python3
"""
API Server for SVG-Browgene

This script creates a FastAPI server that exposes the SVG-Browgene functionality
as REST API endpoints, making it easy to integrate with other applications.
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
import base64
import io
from PIL import Image
from pydantic import BaseModel, Field
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

# Import Playwright for script mode (always available)
from playwright.async_api import async_playwright

# Try to import SVG-Browgene components for AI mode (optional)
BROWGENE_AI_AVAILABLE = False
try:
    from langchain_openai import ChatOpenAI
    from browser_use import Agent, Browser, BrowserConfig
    from src.agent.custom_agent import CustomAgent
    from src.browser.custom_browser import CustomBrowser
    from src.utils.utils import capture_screenshot
    BROWGENE_AI_AVAILABLE = True
except ImportError as e:
    logging.warning(f"BrowGene AI mode not available: {e}. Script mode will still work.")

# Create a utils module for helper functions
class utils:
    @staticmethod
    def get_current_time_iso():
        from datetime import datetime
        return datetime.now().isoformat()
        
    @staticmethod
    def get_llm_model(provider, model_name, num_ctx=4096, temperature=0.7, base_url=None, api_key=None):
        # Import the full LLM model function from utils
        from src.utils.utils import get_llm_model
        return get_llm_model(
            provider=provider,
            model_name=model_name,
            num_ctx=num_ctx,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key
        )
            
    @staticmethod
    async def capture_screenshot_direct(browser_context):
        """Capture a screenshot directly using Playwright"""
        try:
            # Get the browser instance
            if not browser_context or not hasattr(browser_context, "browser"):
                logger.error("Invalid browser context for screenshot")
                return None
                
            # Get the playwright browser
            playwright_browser = browser_context.browser.playwright_browser
            if not playwright_browser:
                logger.error("No playwright browser available")
                return None
                
            # Get the context
            if not playwright_browser.contexts:
                logger.error("No browser contexts available")
                return None
                
            playwright_context = playwright_browser.contexts[0]
            
            # Get the pages
            if not playwright_context.pages:
                logger.error("No pages available in context")
                return None
                
            # Find an active page
            active_page = None
            for page in playwright_context.pages:
                if page.url != "about:blank":
                    active_page = page
                    break
                    
            if not active_page and playwright_context.pages:
                active_page = playwright_context.pages[0]
                
            if not active_page:
                logger.error("No active page found for screenshot")
                return None
                
            # Take the screenshot
            screenshot_bytes = await active_page.screenshot(type="png")
            
            # Encode to base64
            encoded = base64.b64encode(screenshot_bytes).decode("utf-8")
            logger.info(f"Successfully captured screenshot, length: {len(encoded)}")
            return encoded
        except Exception as e:
            logger.error(f"Error capturing screenshot: {e}", exc_info=True)
            return None


# Load environment variables
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="BrowGene API",
    version="1.0.0",
    description="API for browser automation with AI agents using LLMs and Playwright.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker"""
    return {"status": "healthy", "service": "browgene-v2"}

# Define request models
class BrowserOptions(BaseModel):
    """Browser launch options for Playwright"""
    ignoreDefaultArgs: Optional[List[str]] = Field(
        default_factory=list,
        description="List of default browser arguments to ignore"
    )
    args: Optional[List[str]] = Field(
        default_factory=list,
        description="Additional browser arguments to pass to Playwright"
    )

class TaskRequest(BaseModel):
    """Request model for running a browser automation task"""
    task: str = Field(
        ...,
        description="The task description for the AI agent to perform"
    )
    agent_type: str = Field(
        "org",
        description="Type of agent to use (currently only 'org' is supported)"
    )
    llm_provider: str = Field(
        "openai",
        description="LLM provider to use (openai, anthropic, etc.)"
    )
    llm_model_name: str = Field(
        "gpt-4o",
        description="Model name to use for the LLM"
    )
    llm_num_ctx: int = Field(
        4096,
        description="Context window size for the LLM"
    )
    llm_temperature: float = Field(
        0.7,
        description="Temperature parameter for the LLM (0.0-1.0)"
    )
    llm_base_url: Optional[str] = Field(
        None,
        description="Base URL for the LLM API (optional)"
    )
    llm_api_key: Optional[str] = Field(
        None,
        description="API key for the LLM provider (optional, will use env var if not provided)"
    )
    use_own_browser: bool = Field(
        False,
        description="Whether to use an existing browser instance (not recommended)"
    )
    headless: bool = Field(
        True,
        description="Whether to run the browser in headless mode"
    )
    disable_security: bool = Field(
        True,
        description="Whether to disable browser security features"
    )
    use_vision: bool = Field(
        True,
        description="Whether to use vision capabilities of the LLM"
    )
    max_steps: int = Field(
        100,
        description="Maximum number of steps for the agent to take"
    )
    max_actions_per_step: int = Field(
        10,
        description="Maximum number of actions per step"
    )
    window_w: int = Field(
        1280,
        description="Width of the browser window"
    )
    window_h: int = Field(
        720,
        description="Height of the browser window"
    )
    enable_recording: bool = Field(
        False,
        description="Whether to enable recording of the browser session"
    )
    save_recording_path: str = Field(
        "./recordings",
        description="Path to save recordings to"
    )
    save_agent_history_path: str = Field(
        "./agent_history",
        description="Path to save agent history to"
    )
    save_trace_path: str = Field(
        "./traces",
        description="Path to save traces to"
    )
    tool_calling_method: str = Field(
        "function_calling",
        description="Method to use for tool calling (function_calling or json)"
    )
    add_infos: Optional[str] = Field(
        None,
        description="Additional information as JSON string (e.g., browser_options)"
    )

# Define response models
class TaskResponse(BaseModel):
    """Response model for browser automation tasks"""
    success: bool = Field(
        ...,
        description="Whether the task was successful"
    )
    extracted_text: str = Field(
        "",
        description="Extracted text from the task"
    )
    interactions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of interactions between the agent and the browser"
    )
    agent_brain: Dict[str, Any] = Field(
        default_factory=dict,
        description="Agent brain state"
    )
    history_file: Optional[str] = Field(
        None,
        description="Path to the agent history file"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if the task failed"
    )
    screenshots: List[str] = Field(
        default_factory=list,
        description="List of base64-encoded screenshots captured during task execution"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the task execution"
    )

# Store active tasks
active_tasks = {}

@app.get("/", summary="Root Endpoint", description="Check if the API server is running")
def root():
    """
    Returns the status of the API server.
    
    Returns:
        dict: A message indicating the server is running
    """
    return {"message": "SVG-Browgene API Server is running"}

@app.post("/api/browgene/run_task", response_model=TaskResponse,
         summary="Run Task",
         description="Run a browser automation task using AI agents")
async def run_task(request: TaskRequest, background_tasks: BackgroundTasks):
    """
    Run a browser automation task using AI agents and Playwright.
    
    The task will be executed asynchronously in the background. The response will
    contain a task ID that can be used to check the status of the task.
    
    Args:
        request: The task request containing all parameters
        background_tasks: FastAPI background tasks handler
        
    Returns:
        TaskResponse: Initial response with task ID and status
    """
    # Generate a task ID
    task_id = f"task_{len(active_tasks) + 1}"
    
    # Initialize response
    response = TaskResponse(
        success=False,
        metadata={
            "task_id": task_id,
            "task": request.task,
            "status": "pending",
            "start_time": utils.get_current_time_iso()
        }
    )
    
    # Store in active tasks
    active_tasks[task_id] = response
    
    try:
        # Parse browser options from add_infos if provided
        browser_options = {}
        if request.add_infos:
            try:
                browser_options = json.loads(request.add_infos)
            except json.JSONDecodeError:
                logger.warning("Failed to parse add_infos as JSON, using empty dict")
        
        # Get API key from environment or request
        api_key = request.llm_api_key or os.environ.get(f"{request.llm_provider.upper()}_API_KEY")
        if not api_key and request.llm_provider == "openai":
            raise ValueError("OpenAI API key not found")
        
        # Initialize LLM
        llm = utils.get_llm_model(
            provider=request.llm_provider,
            model_name=request.llm_model_name,
            num_ctx=request.llm_num_ctx,
            temperature=request.llm_temperature,
            base_url=request.llm_base_url,
            api_key=api_key
        )
        
        # Start task execution in background
        background_tasks.add_task(
            execute_task,
            task_id=task_id,
            request=request,
            llm=llm,
            browser_options=browser_options
        )
        
        # Return initial response
        response.success = True
        response.metadata["status"] = "started"
        
        return response
        
    except Exception as e:
        logger.error(f"Error starting task: {e}")
        response.error = str(e)
        return response

@app.get("/api/browgene/task_status/{task_id}", response_model=TaskResponse,
       summary="Get Task Status",
       description="Check the current status of a previously started task")
async def get_task_status(task_id: str):
    """
    Get the status of a running task.
    
    This endpoint allows you to check the current status of a task that was
    previously started using the run_task endpoint. The response will contain
    the current status of the task, as well as any results if the task has completed.
    
    Args:
        task_id: The ID of the task to check
        
    Returns:
        TaskResponse: Current status and results of the task
        
    Raises:
        HTTPException: If the task ID is not found
    """
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return active_tasks[task_id]

async def execute_task(task_id: str, request: TaskRequest, llm: Any, browser_options: Dict[str, Any]):
    """
    Execute a browser automation task
    """
    # Initialize response
    response = TaskResponse(
        success=False,
        metadata={
            "task_id": task_id,
            "task": request.task,
            "status": "running",
            "start_time": utils.get_current_time_iso()
        }
    )
    
    # Store in active tasks
    active_tasks[task_id] = response
    
    try:
        # Configure browser launch options with Docker-specific arguments
        docker_chromium_args = [
            "--no-sandbox",
            "--disable-setuid-sandbox", 
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-gpu-sandbox",
            "--disable-software-rasterizer",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--disable-field-trial-config",
            "--disable-back-forward-cache",
            "--disable-ipc-flooding-protection",
            "--disable-extensions",
            "--disable-plugins",
            "--disable-web-security",
            "--disable-features=VizDisplayCompositor,TranslateUI,BlinkGenPropertyTrees",
            "--enable-automation",
            "--force-color-profile=srgb",
            "--disable-background-networking",
            "--disable-default-apps",
            "--disable-sync",
            "--no-default-browser-check",
            "--no-first-run",
            "--disable-gpu-process-crash-limit",
            "--single-process",
            "--remote-debugging-port=9222",
            "--disable-blink-features=AutomationControlled",
            "--disable-client-side-phishing-detection",
            "--disable-component-update",
            "--disable-domain-reliability",
            "--disable-hang-monitor",
            "--disable-prompt-on-repost",
            "--no-proxy-server",
            "--ignore-certificate-errors",
            "--ignore-ssl-errors",
            "--ignore-certificate-errors-spki-list",
            "--ignore-certificate-errors-ssl-errors",
            "--disable-dev-tools",
            "--disable-logging",
            "--disable-breakpad",
            "--disable-crash-reporter",
            "--no-crash-upload",
            "--disable-background-mode",
            "--disable-default-apps",
            "--disable-translate",
            "--disable-popup-blocking",
            "--allow-running-insecure-content",
            "--disable-features=VizDisplayCompositor,VizHitTestSurfaceLayer"
        ]
        
        # Create browser with proper configuration and Docker-specific args
        logger.info("Creating new browser instance with Docker-specific arguments...")
        browser = Browser(
            config=BrowserConfig(
                disable_security=request.disable_security,
                headless=request.headless,
                extra_chromium_args=docker_chromium_args
            )
        )
        logger.info("Browser instance created successfully")
        
        # Create window size configuration
        window_size = {
            "width": request.window_w,
            "height": request.window_h
        }
        
        # Create directories if they don't exist
        os.makedirs(request.save_recording_path, exist_ok=True)
        os.makedirs(request.save_agent_history_path, exist_ok=True)
        os.makedirs(request.save_trace_path, exist_ok=True)
        
        # Create browser context with proper configuration
        logger.info("Creating new browser context...")
        browser_context = await browser.new_context(
            config=BrowserContextConfig(
                trace_path=request.save_trace_path if request.save_trace_path else None,
                save_recording_path=request.save_recording_path if request.enable_recording else None,
                no_viewport=False,
                browser_window_size=BrowserContextWindowSize(
                    width=request.window_w, 
                    height=request.window_h
                )
            )
        )
        logger.info("Browser context created successfully")
        
        # Initialize agent based on agent type
        if request.agent_type == "org":
            # Temporarily use standard Agent for testing connection issues
            agent = Agent(
                task=request.task,
                llm=llm,
                browser=browser,
                browser_context=browser_context,
                use_vision=request.use_vision
            )
        else:
            # Default to custom agent
            # Prepare agent parameters - IMPORTANT: Do not include include_attributes parameter
            agent_params = {
                "task": request.task,
                "llm": llm,
                "browser": browser,
                "browser_context": browser_context,
                "use_vision": request.use_vision,
                "max_actions_per_step": request.max_actions_per_step,
                # Do not include include_attributes here
            }
            
            # Add optional parameters
            if request.enable_recording:
                agent_params["generate_gif"] = True
            
            # Use save_conversation_path instead of save_agent_history_path
            if request.save_agent_history_path:
                agent_params["save_conversation_path"] = request.save_agent_history_path
                
            # Create agent with filtered parameters
            agent = CustomAgent(**agent_params)
        
        # Run the agent with max_steps parameter
        output = await agent.run(max_steps=request.max_steps)
        
        # Extract the final result/answer from the agent
        extracted_text = ""
        try:
            # Try to get the final answer from the agent's message manager
            if hasattr(agent, 'message_manager') and hasattr(agent.message_manager, 'get_messages'):
                messages = agent.message_manager.get_messages()
                # Look for AI messages that contain task completion or results
                for msg in reversed(messages):
                    if hasattr(msg, 'content') and msg.content:
                        content_str = ""
                        if isinstance(msg.content, str):
                            content_str = msg.content
                        elif isinstance(msg.content, list):
                            # Handle structured content
                            text_parts = []
                            for part in msg.content:
                                if isinstance(part, dict) and part.get('type') == 'text':
                                    text_parts.append(part.get('text', ''))
                            content_str = '\n'.join(text_parts)
                        
                        # Look for content that contains task completion or results
                        if content_str and any(keyword in content_str.lower() for keyword in [
                            'done', 'completed', 'summary', 'found', 'information about', 
                            'heather bagg', 'obituary', 'result', 'gathered'
                        ]):
                            # Skip system prompts and instructions
                            if not any(skip_phrase in content_str for skip_phrase in [
                                'You are a precise browser automation agent',
                                'RESPONSE FORMAT',
                                'INPUT STRUCTURE',
                                'Functions:'
                            ]):
                                extracted_text = content_str
                                break
            
            # If no specific result found, try to extract from agent history
            if not extracted_text and hasattr(agent, 'history') and agent.history:
                try:
                    # Check if history has completion information
                    if hasattr(agent.history, 'history'):
                        history_items = agent.history.history
                    else:
                        history_items = agent.history
                    
                    # Look for the last meaningful result
                    for item in reversed(history_items):
                        if hasattr(item, 'result') and item.result:
                            result_str = str(item.result)
                            if any(keyword in result_str.lower() for keyword in [
                                'heather bagg', 'obituary', 'summary', 'information'
                            ]):
                                extracted_text = result_str
                                break
                except Exception as hist_e:
                    logger.error(f"Error processing history: {hist_e}")
            
            # Fallback: try to extract from the output object
            if not extracted_text and output:
                extracted_text = str(output)
                
        except Exception as e:
            logger.error(f"Error extracting final result: {e}")
            extracted_text = str(output) if output else ""
        
        # Update response
        response.success = True
        response.extracted_text = extracted_text
        
        # Get agent history if available
        if hasattr(agent, "history") and agent.history:
            try:
                # Try to process history as a list of objects with action/result attributes
                interactions = []
                screenshots = []
                
                # Check if history is an object with a history attribute (AgentHistoryList)
                if hasattr(agent.history, "history"):
                    history_items = agent.history.history
                else:
                    history_items = agent.history
                    
                for step in history_items:
                    # Check if step is a tuple or has action/result attributes
                    if isinstance(step, tuple) and len(step) >= 2:
                        interactions.append({
                            "action": str(step[0]),
                            "result": str(step[1])
                        })
                    elif hasattr(step, "action") and hasattr(step, "result"):
                        interactions.append({
                            "action": str(step.action),
                            "result": str(step.result)
                        })
                    else:
                        # Fallback for unknown history format
                        interactions.append({
                            "action": str(step),
                            "result": ""
                        })
                    
                    # Extract screenshots from history if available
                    if hasattr(step, "state") and hasattr(step.state, "screenshot") and step.state.screenshot:
                        screenshots.append(step.state.screenshot)
                        
                response.interactions = interactions
                response.screenshots = screenshots
                
                # Always try to capture a fresh screenshot regardless of history
                try:
                    logger.info("Attempting to capture a fresh screenshot using direct method")
                    current_screenshot = await utils.capture_screenshot_direct(browser_context)
                    if current_screenshot:
                        logger.info(f"Successfully captured screenshot of length: {len(current_screenshot)}")
                        response.screenshots = [current_screenshot]
                    else:
                        # Try the regular method as fallback
                        logger.warning("Direct screenshot capture failed, trying fallback method")
                        fallback_screenshot = await capture_screenshot(browser_context)
                        if fallback_screenshot:
                            logger.info(f"Fallback screenshot captured, length: {len(fallback_screenshot)}")
                            response.screenshots = [fallback_screenshot]
                        else:
                            logger.warning("Both screenshot methods returned None")
                except Exception as screenshot_error:
                    logger.error(f"Error capturing screenshot: {screenshot_error}", exc_info=True)
                        
            except Exception as e:
                logger.error(f"Error processing agent history: {e}")
                # Provide a simple fallback
                response.interactions = [{"action": "Task execution", "result": str(output)}]
        
        # Get agent brain if available
        if hasattr(agent, "brain") and agent.brain:
            # Convert to dict if needed
            if hasattr(agent.brain, "__dict__"):
                response.agent_brain = agent.brain.__dict__
            else:
                response.agent_brain = {"data": str(agent.brain)}
        
        # Get history file if available
        if hasattr(agent, "history_file") and agent.history_file:
            response.history_file = agent.history_file
        
        # Update metadata
        response.metadata["status"] = "completed"
        response.metadata["end_time"] = utils.get_current_time_iso()
        
    except Exception as e:
        logger.error(f"Error executing task: {e}", exc_info=True)
        response.error = str(e)
        response.metadata["status"] = "failed"
        response.metadata["end_time"] = utils.get_current_time_iso()
    finally:
        # Clean up browser and context
        try:
            if 'browser_context' in locals() and browser_context:
                logger.info("Closing browser context...")
                await browser_context.close()
            if 'browser' in locals() and browser:
                logger.info("Closing browser...")
                await browser.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        # Update active tasks
        active_tasks[task_id] = response

# ============================================================================
# SCRIPT MODE - Pure Playwright automation (no LLM required)
# ============================================================================

class ScriptAction(BaseModel):
    """A single action to perform in script mode"""
    type: str = Field(
        ...,
        description="Action type: goto, fill, click, wait_for, extract, screenshot, select, check, uncheck, type, press, scroll, evaluate"
    )
    selector: Optional[str] = Field(
        None,
        description="CSS or XPath selector for the target element"
    )
    value: Optional[str] = Field(
        None,
        description="Value for fill/type/select actions, or URL for goto"
    )
    url: Optional[str] = Field(
        None,
        description="URL for goto action"
    )
    timeout: Optional[int] = Field(
        30000,
        description="Timeout in milliseconds for wait operations"
    )
    outputAs: Optional[str] = Field(
        None,
        description="Variable name to store extracted data or screenshot"
    )
    script: Optional[str] = Field(
        None,
        description="JavaScript code for evaluate action"
    )
    key: Optional[str] = Field(
        None,
        description="Key to press for press action (e.g., 'Enter', 'Tab')"
    )
    direction: Optional[str] = Field(
        "down",
        description="Scroll direction: up, down, left, right"
    )
    amount: Optional[int] = Field(
        500,
        description="Scroll amount in pixels"
    )
    attribute: Optional[str] = Field(
        None,
        description="Attribute to extract (e.g., 'href', 'src'). If None, extracts text content"
    )
    all: Optional[bool] = Field(
        False,
        description="If True, extract from all matching elements (returns array)"
    )


class ScriptRequest(BaseModel):
    """Request model for script-based browser automation"""
    url: Optional[str] = Field(
        None,
        description="Initial URL to navigate to (optional if first action is goto)"
    )
    actions: List[ScriptAction] = Field(
        ...,
        description="List of actions to perform in sequence"
    )
    variables: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Variables for interpolation in action values (e.g., {{username}})"
    )
    headless: bool = Field(
        True,
        description="Whether to run browser in headless mode"
    )
    window_w: int = Field(
        1280,
        description="Browser window width"
    )
    window_h: int = Field(
        720,
        description="Browser window height"
    )
    timeout: int = Field(
        30000,
        description="Default timeout for actions in milliseconds"
    )
    user_agent: Optional[str] = Field(
        None,
        description="Custom user agent string"
    )
    cookies: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Cookies to set before automation"
    )
    headers: Optional[Dict[str, str]] = Field(
        None,
        description="Extra HTTP headers to set"
    )


class ScriptResponse(BaseModel):
    """Response model for script-based automation"""
    success: bool = Field(
        ...,
        description="Whether all actions completed successfully"
    )
    results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted data and screenshots keyed by outputAs names"
    )
    actions_completed: int = Field(
        0,
        description="Number of actions successfully completed"
    )
    actions_total: int = Field(
        0,
        description="Total number of actions in the script"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if script failed"
    )
    failed_action: Optional[int] = Field(
        None,
        description="Index of the action that failed (0-based)"
    )
    execution_time_ms: int = Field(
        0,
        description="Total execution time in milliseconds"
    )
    final_url: Optional[str] = Field(
        None,
        description="Final URL after all actions"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the execution"
    )


def interpolate_variables(text: str, variables: Dict[str, Any]) -> str:
    """Replace {{variable}} placeholders with actual values"""
    if not text or not variables:
        return text
    
    import re
    result = text
    
    # Find all {{variable}} patterns
    pattern = r'\{\{([^}]+)\}\}'
    matches = re.findall(pattern, text)
    
    for match in matches:
        var_name = match.strip()
        if var_name in variables:
            value = variables[var_name]
            result = result.replace(f'{{{{{match}}}}}', str(value))
    
    return result


@app.post("/api/browgene/run_script", response_model=ScriptResponse,
         summary="Run Script",
         description="Run deterministic browser automation using Playwright (no LLM required)")
async def run_script(request: ScriptRequest):
    """
    Execute a sequence of browser actions using pure Playwright.
    
    This endpoint provides deterministic, fast browser automation without
    requiring an LLM. Ideal for login flows, form filling, and data extraction
    with known page structures.
    
    Supported action types:
    - goto: Navigate to a URL
    - fill: Fill an input field
    - click: Click an element
    - wait_for: Wait for an element to appear
    - extract: Extract text/attribute from element(s)
    - screenshot: Capture a screenshot
    - select: Select a dropdown option
    - check/uncheck: Toggle checkboxes
    - type: Type text with keyboard events
    - press: Press a keyboard key
    - scroll: Scroll the page
    - evaluate: Run JavaScript code
    
    Args:
        request: The script request containing actions and configuration
        
    Returns:
        ScriptResponse: Results of the script execution
    """
    import time
    start_time = time.time()
    
    response = ScriptResponse(
        success=False,
        actions_total=len(request.actions),
        metadata={
            "start_time": utils.get_current_time_iso()
        }
    )
    
    playwright = None
    browser = None
    context = None
    page = None
    
    try:
        # Start Playwright
        playwright = await async_playwright().start()
        
        # Launch browser
        browser = await playwright.chromium.launch(
            headless=request.headless,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                f'--window-size={request.window_w},{request.window_h}'
            ]
        )
        
        # Create context with optional settings
        context_options = {
            "viewport": {"width": request.window_w, "height": request.window_h}
        }
        
        if request.user_agent:
            context_options["user_agent"] = request.user_agent
            
        if request.headers:
            context_options["extra_http_headers"] = request.headers
        
        context = await browser.new_context(**context_options)
        
        # Set cookies if provided
        if request.cookies:
            await context.add_cookies(request.cookies)
        
        # Create page
        page = await context.new_page()
        page.set_default_timeout(request.timeout)
        
        # Navigate to initial URL if provided
        if request.url:
            initial_url = interpolate_variables(request.url, request.variables or {})
            await page.goto(initial_url, wait_until="domcontentloaded")
        
        # Execute actions
        results = {}
        
        for i, action in enumerate(request.actions):
            try:
                action_type = action.type.lower()
                selector = interpolate_variables(action.selector, request.variables or {}) if action.selector else None
                value = interpolate_variables(action.value, request.variables or {}) if action.value else None
                
                logger.info(f"Executing action {i+1}/{len(request.actions)}: {action_type}")
                
                if action_type == "goto":
                    url = action.url or value
                    if url:
                        url = interpolate_variables(url, request.variables or {})
                        await page.goto(url, wait_until="domcontentloaded")
                
                elif action_type == "fill":
                    if selector and value is not None:
                        await page.fill(selector, value)
                
                elif action_type == "click":
                    if selector:
                        await page.click(selector)
                
                elif action_type == "wait_for":
                    if selector:
                        timeout = action.timeout or request.timeout
                        await page.wait_for_selector(selector, timeout=timeout)
                
                elif action_type == "extract":
                    logger.info(f"Extract action: selector={selector}, outputAs={action.outputAs}, all={action.all}, attribute={action.attribute}")
                    if selector:
                        if action.all:
                            # Extract from all matching elements
                            elements = await page.query_selector_all(selector)
                            extracted = []
                            for el in elements:
                                if action.attribute:
                                    val = await el.get_attribute(action.attribute)
                                else:
                                    val = await el.text_content()
                                extracted.append(val)
                            
                            logger.info(f"Extracted {len(extracted)} items from all matching elements")
                            if action.outputAs:
                                results[action.outputAs] = extracted
                                logger.info(f"Stored results under key: {action.outputAs}")
                        else:
                            # Extract from first matching element
                            element = await page.query_selector(selector)
                            if element:
                                if action.attribute:
                                    extracted = await element.get_attribute(action.attribute)
                                else:
                                    extracted = await element.text_content()
                                
                                if action.outputAs:
                                    results[action.outputAs] = extracted
                
                elif action_type == "screenshot":
                    screenshot_bytes = await page.screenshot(type="png", full_page=False)
                    screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")
                    
                    output_name = action.outputAs or f"screenshot_{i}"
                    results[output_name] = screenshot_base64
                
                elif action_type == "select":
                    if selector and value:
                        await page.select_option(selector, value)
                
                elif action_type == "check":
                    if selector:
                        await page.check(selector)
                
                elif action_type == "uncheck":
                    if selector:
                        await page.uncheck(selector)
                
                elif action_type == "type":
                    if selector and value is not None:
                        await page.type(selector, value)
                
                elif action_type == "press":
                    key = action.key or value
                    if key:
                        if selector:
                            await page.press(selector, key)
                        else:
                            await page.keyboard.press(key)
                
                elif action_type == "scroll":
                    direction = action.direction or "down"
                    amount = action.amount or 500
                    
                    if direction == "down":
                        await page.evaluate(f"window.scrollBy(0, {amount})")
                    elif direction == "up":
                        await page.evaluate(f"window.scrollBy(0, -{amount})")
                    elif direction == "right":
                        await page.evaluate(f"window.scrollBy({amount}, 0)")
                    elif direction == "left":
                        await page.evaluate(f"window.scrollBy(-{amount}, 0)")
                
                elif action_type == "evaluate":
                    script = action.script or value
                    if script:
                        script = interpolate_variables(script, request.variables or {})
                        result = await page.evaluate(script)
                        
                        if action.outputAs:
                            results[action.outputAs] = result
                
                elif action_type == "wait":
                    # Simple wait/delay
                    delay_ms = action.timeout or 1000
                    await asyncio.sleep(delay_ms / 1000)
                
                else:
                    logger.warning(f"Unknown action type: {action_type}")
                
                response.actions_completed = i + 1
                
            except Exception as action_error:
                logger.error(f"Action {i+1} failed: {action_error}")
                response.error = f"Action {i+1} ({action.type}) failed: {str(action_error)}"
                response.failed_action = i
                break
        
        # Get final URL
        response.final_url = page.url
        
        # Check if all actions completed
        if response.actions_completed == len(request.actions):
            response.success = True
        
        response.results = results
        
    except Exception as e:
        logger.error(f"Script execution error: {e}", exc_info=True)
        response.error = str(e)
        
    finally:
        # Cleanup
        try:
            if page:
                await page.close()
            if context:
                await context.close()
            if browser:
                await browser.close()
            if playwright:
                await playwright.stop()
        except Exception as cleanup_error:
            logger.error(f"Cleanup error: {cleanup_error}")
        
        # Calculate execution time
        response.execution_time_ms = int((time.time() - start_time) * 1000)
        response.metadata["end_time"] = utils.get_current_time_iso()
    
    return response


@app.get("/api/browgene/health", summary="Health Check", description="Check API server health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "svg-browgene",
        "version": "1.0.0",
        "endpoints": {
            "ai_mode": "/api/browgene/run_task",
            "script_mode": "/api/browgene/run_script",
            "task_status": "/api/browgene/task_status/{task_id}"
        }
    }


def start_server():
    """Start the API server"""
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=7793,
        reload=True
    )

if __name__ == "__main__":
    start_server()
