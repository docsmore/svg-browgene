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

# Import SVG-Browgene components
from langchain_openai import ChatOpenAI
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig, BrowserContextWindowSize
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from playwright.async_api import async_playwright
from src.utils.utils import capture_screenshot

# Create a utils module for helper functions
class utils:
    @staticmethod
    def get_current_time_iso():
        from datetime import datetime
        return datetime.now().isoformat()
        
    @staticmethod
    def get_llm_model(provider, model_name, num_ctx=4096, temperature=0.7, base_url=None, api_key=None):
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                openai_api_key=api_key,
                openai_api_base=base_url
            )
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model_name=model_name,
                temperature=temperature,
                anthropic_api_key=api_key
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
            
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
    title="SVG-Browgene API",
    description="API for browser automation with AI agents using LLMs and Playwright.",
    version="1.0.0",
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
        # Initialize playwright and browser
        playwright = await async_playwright().start()
        
        # Configure browser launch options
        browser_args = browser_options.get("args", [])
        ignore_default_args = browser_options.get("ignoreDefaultArgs", [])
        
        browser_instance = await playwright.chromium.launch(
            headless=request.headless,
            args=browser_args,
            ignore_default_args=ignore_default_args
        )
        
        # Create browser with proper configuration
        browser = Browser(
            config=BrowserConfig(
                disable_security=request.disable_security,
                headless=request.headless
            )
        )
        
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
        
        # Initialize agent based on agent type
        if request.agent_type == "org":
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
        
        # Update response
        response.success = True
        response.extracted_text = str(output) if output else ""
        
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
        # Clean up
        try:
            if 'browser' in locals():
                await browser.close()
            if 'playwright' in locals():
                await playwright.stop()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        # Update active tasks
        active_tasks[task_id] = response

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
