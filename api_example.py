#!/usr/bin/env python3
"""
API Example for SVG-Browgene

This script demonstrates how to use the SVG-Browgene API to run browser automation tasks
programmatically without using the Gradio UI.
"""

import os
import asyncio
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from browser_use.browser.browser import Browser
from browser_use.browser.config import BrowserConfig
from browser_use.browser.context import BrowserContextConfig, BrowserContextWindowSize
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from playwright.async_api import async_playwright

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def run_browser_task(task_description, options=None):
    """
    Run a browser automation task using the SVG-Browgene API
    
    Args:
        task_description (str): The task to perform
        options (dict, optional): Configuration options
    
    Returns:
        dict: Results of the task execution
    """
    if options is None:
        options = {}
    
    # Default options
    default_options = {
        "llm_provider": "openai",
        "llm_model": "gpt-4o",
        "temperature": 1.0,
        "headless": False,
        "window_width": 1280,
        "window_height": 720,
        "max_steps": 15,
        "use_vision": True,
        "record_session": True,
        "recording_path": "./recordings",
        "history_path": "./history"
    }
    
    # Merge options
    for key, value in default_options.items():
        if key not in options:
            options[key] = value
    
    # Check if OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")
    
    logger.info(f"Running task: {task_description}")
    
    # Initialize LLM based on provider
    if options["llm_provider"] == "openai":
        llm = ChatOpenAI(
            model_name=options["llm_model"],
            temperature=options["temperature"],
            openai_api_key=api_key
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {options['llm_provider']}")
    
    # Initialize browser with playwright
    playwright = await async_playwright().start()
    browser_instance = await playwright.chromium.launch(headless=options["headless"])
    
    # Create browser config
    browser_config = BrowserConfig(
        disable_security=options.get("disable_security", False)
    )
    
    # Create browser
    browser = Browser(browser_instance, config=browser_config)
    
    # Create window size configuration
    window_size = BrowserContextWindowSize(
        width=options["window_width"],
        height=options["window_height"]
    )
    
    # Create browser context config
    context_config = BrowserContextConfig(
        window_size=window_size
    )
    
    results = {
        "success": False,
        "output": "",
        "error": None,
        "history_file": None,
        "recording_file": None
    }
    
    try:
        # Create a new browser context
        browser_context = await browser.new_context(config=context_config)
        
        # Create directories if they don't exist
        os.makedirs(options["recording_path"], exist_ok=True)
        os.makedirs(options["history_path"], exist_ok=True)
        
        # Initialize agent
        agent = CustomAgent(
            task=task_description,
            llm=llm,
            browser=browser,
            browser_context=browser_context,
            use_vision=options["use_vision"],
            max_steps=options["max_steps"],
            generate_gif=options["record_session"],
            save_recording_path=options["recording_path"] if options["record_session"] else None,
            save_agent_history_path=options["history_path"]
        )
        
        # Run the agent
        output = await agent.run()
        
        # Get results
        results["success"] = True
        results["output"] = output
        
        # Get history file if available
        if agent.history_file:
            results["history_file"] = agent.history_file
        
        # Get recording file if available
        if options["record_session"] and agent.history_gif_path:
            results["recording_file"] = agent.history_gif_path
        
        logger.info("Task completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running task: {e}")
        results["error"] = str(e)
    finally:
        # Clean up
        await browser.close()
        await playwright.stop()
    
    return results

async def main():
    # Example task
    task = "Search for 'weather in New York' on Google and tell me the current temperature"
    
    # Run the task with default options
    results = await run_browser_task(task)
    
    # Print results
    if results["success"]:
        print(f"Task output: {results['output']}")
        if results["history_file"]:
            print(f"History saved to: {results['history_file']}")
        if results["recording_file"]:
            print(f"Recording saved to: {results['recording_file']}")
    else:
        print(f"Task failed: {results['error']}")

    # Example with custom options
    custom_task = "Go to amazon.com and find the top-rated wireless headphones under $100"
    custom_options = {
        "llm_model": "gpt-4o",
        "headless": False,
        "max_steps": 20,
        "window_width": 1600,
        "window_height": 900
    }
    
    print("\nRunning custom task...")
    custom_results = await run_browser_task(custom_task, custom_options)
    
    # Print custom results
    if custom_results["success"]:
        print(f"Custom task output: {custom_results['output']}")
    else:
        print(f"Custom task failed: {custom_results['error']}")

if __name__ == "__main__":
    asyncio.run(main())
