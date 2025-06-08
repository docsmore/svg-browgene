#!/usr/bin/env python3
"""
Simple script to run a browser automation task using the SVG-Browgene core functionality.
This bypasses the Gradio UI which is having compatibility issues.
"""

import os
import asyncio
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from playwright.async_api import async_playwright
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContextConfig
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def run_browser_task(task_description):
    """Run a browser automation task using the CustomAgent"""
    
    # Check if OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")
        return
    
    logger.info(f"Running task: {task_description}")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=1.0,
        openai_api_key=api_key
    )
    
    # Initialize playwright and browser
    playwright = await async_playwright().start()
    browser_instance = await playwright.chromium.launch(headless=False)
    
    # Create browser with proper configuration
    from browser_use.browser.config import BrowserConfig
    browser_config = BrowserConfig()
    browser = Browser(browser_instance, config=browser_config)
    
    try:
        # Create a new browser context
        browser_context = await browser.new_context()
        
        # Initialize agent
        agent = CustomAgent(
            task=task_description,
            llm=llm,
            browser=browser,
            browser_context=browser_context,
            use_vision=True,
            max_steps=10,  # Limit to 10 steps for testing
            generate_gif=True  # Save a GIF of the browser interaction
        )
        
        # Run the agent
        await agent.run()
        
        logger.info("Task completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running task: {e}")
    finally:
        # Clean up
        await browser.close()

async def main():
    # Example task
    task = "Search for 'weather in New York' on Google and tell me the current temperature"
    
    # You can replace with your own task
    await run_browser_task(task)

if __name__ == "__main__":
    asyncio.run(main())
