#!/usr/bin/env python3
"""
Script to explore the SVG-Browgene project structure and components.
This script doesn't run the browser automation but helps understand the project.
"""

import os
import sys
import inspect
import importlib
from pprint import pprint

def print_section(title):
    """Print a section title with formatting"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def explore_module(module_name):
    """Explore and print information about a module"""
    try:
        module = importlib.import_module(module_name)
        print(f"Module: {module_name}")
        
        # Get classes and functions
        classes = []
        functions = []
        
        for name, obj in inspect.getmembers(module):
            if name.startswith('_'):
                continue
                
            if inspect.isclass(obj):
                classes.append(name)
            elif inspect.isfunction(obj):
                functions.append(name)
        
        if classes:
            print("\nClasses:")
            for cls in classes:
                print(f"  - {cls}")
                
        if functions:
            print("\nFunctions:")
            for func in functions:
                print(f"  - {func}")
                
        return True
    except ImportError as e:
        print(f"Could not import module {module_name}: {e}")
        return False
    except Exception as e:
        print(f"Error exploring module {module_name}: {e}")
        return False

def explore_project_structure():
    """Explore the project structure"""
    print_section("Project Structure")
    
    # List main directories
    src_dir = os.path.join(os.getcwd(), "src")
    if os.path.exists(src_dir):
        print("Source directories:")
        for item in os.listdir(src_dir):
            item_path = os.path.join(src_dir, item)
            if os.path.isdir(item_path) and not item.startswith('__'):
                print(f"  - src/{item}")
    
    # List main Python files
    print("\nMain Python files:")
    for item in os.listdir(os.getcwd()):
        if item.endswith('.py') and os.path.isfile(item):
            print(f"  - {item}")

def explore_agent_module():
    """Explore the agent module"""
    print_section("Agent Module")
    explore_module("src.agent.custom_agent")
    
def explore_browser_module():
    """Explore the browser module"""
    print_section("Browser Module")
    explore_module("src.browser.custom_browser")
    
def explore_controller_module():
    """Explore the controller module"""
    print_section("Controller Module")
    explore_module("src.controller.custom_controller")

def explore_environment_variables():
    """Explore environment variables used by the project"""
    print_section("Environment Variables")
    
    env_vars = {
        "API Keys": [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "DEEPSEEK_API_KEY"
        ],
        "Browser Settings": [
            "CHROME_PATH",
            "CHROME_USER_DATA",
            "CHROME_PERSISTENT_SESSION"
        ],
        "Display Settings": [
            "RESOLUTION",
            "RESOLUTION_WIDTH",
            "RESOLUTION_HEIGHT"
        ],
        "Other Settings": [
            "VNC_PASSWORD",
            "BROWSER_USE_LOGGING_LEVEL",
            "ANONYMIZED_TELEMETRY"
        ]
    }
    
    for category, vars in env_vars.items():
        print(f"\n{category}:")
        for var in vars:
            value = os.environ.get(var, "Not set")
            if "API_KEY" in var and value != "Not set":
                # Mask API keys
                value = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
            print(f"  - {var}: {value}")

def explore_webui():
    """Explore the web UI structure"""
    print_section("Web UI Structure")
    
    try:
        import gradio as gr
        print("Gradio version:", gr.__version__)
    except ImportError:
        print("Gradio is not installed or cannot be imported")
    
    # Try to analyze webui.py without importing it
    webui_path = os.path.join(os.getcwd(), "webui.py")
    if os.path.exists(webui_path):
        print("\nAnalyzing webui.py structure:")
        
        with open(webui_path, 'r') as f:
            content = f.read()
            
        # Count functions
        import re
        functions = re.findall(r'def\s+([a-zA-Z0-9_]+)\s*\(', content)
        print(f"  - Found {len(functions)} functions")
        print("  - Main functions:")
        for func in functions[:10]:  # Show first 10
            print(f"    * {func}")
            
        # Check for UI tabs
        tabs = re.findall(r'with\s+gr\.Tab\(["\']([^"\']+)["\']', content)
        if tabs:
            print("\n  - UI Tabs:")
            for tab in tabs:
                print(f"    * {tab}")

def main():
    """Main function to explore the project"""
    print_section("SVG-Browgene Project Explorer")
    
    explore_project_structure()
    explore_agent_module()
    explore_browser_module()
    explore_controller_module()
    explore_environment_variables()
    explore_webui()
    
    print_section("How to Run a Task")
    print("""
To run a task in SVG-Browgene, you would typically:

1. Set up your environment variables (API keys, browser settings)
2. Run the web UI: python webui.py
3. In the web UI:
   - Select the LLM provider
   - Configure browser settings
   - Enter your task description
   - Click "Run" to start the browser automation

Due to compatibility issues with the current Gradio version, 
you may need to use Docker to run the application:

docker compose up

This will start the application in a container with all dependencies properly configured.
You can then access the web UI at http://localhost:7788
    """)

if __name__ == "__main__":
    main()
