import gradio as gr
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def display_info():
    """Display information about the SVG-Browgene project"""
    info = """
    # SVG-Browgene Project Overview
    
    This is a simplified interface to explore the SVG-Browgene project.
    
    ## Project Description
    SVG-Browgene is a browser automation tool that allows AI agents to interact with web browsers.
    It's built on the browser-use framework and provides a web UI for configuring and running browser automation tasks.
    
    ## Key Features
    - AI-Powered Browser Automation
    - Multiple LLM Support (OpenAI, Google, Anthropic, DeepSeek, etc.)
    - Custom Browser Support
    - Screen Recording
    - Persistent Browser Sessions
    
    ## Configuration Options
    - LLM Provider: OpenAI, Google, Anthropic, DeepSeek, etc.
    - Browser Settings: Use own browser, persistent sessions
    - Recording Options: Save recordings, traces
    """
    return info

def show_env_vars():
    """Display the current environment variables"""
    env_vars = {}
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", 
                "CHROME_PATH", "CHROME_USER_DATA", "CHROME_PERSISTENT_SESSION"]:
        value = os.getenv(key, "Not set")
        if key.endswith("API_KEY") and value != "Not set":
            # Mask API keys for security
            value = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
        env_vars[key] = value
    
    result = "## Environment Variables\n\n"
    for key, value in env_vars.items():
        result += f"- **{key}**: {value}\n"
    
    return result

def show_project_structure():
    """Display the project structure"""
    structure = """
    ## Project Structure
    
    - **webui.py**: Main application file
    - **src/**: Core functionality
      - **agent/**: Agent implementation for browser automation
      - **browser/**: Browser integration code
      - **controller/**: Control logic for the agents
      - **utils/**: Utility functions
    - **assets/**: Static assets
    - **data/**: Data storage
    - **tests/**: Test files
    """
    return structure

# Create the Gradio interface
with gr.Blocks(title="SVG-Browgene Explorer") as demo:
    gr.Markdown("# SVG-Browgene Explorer")
    gr.Markdown("This is a simplified interface to explore the SVG-Browgene project.")
    
    with gr.Tabs():
        with gr.Tab("Project Info"):
            info_output = gr.Markdown(display_info())
        
        with gr.Tab("Environment"):
            env_output = gr.Markdown(show_env_vars())
            refresh_btn = gr.Button("Refresh Environment Variables")
            refresh_btn.click(fn=show_env_vars, outputs=env_output)
        
        with gr.Tab("Project Structure"):
            structure_output = gr.Markdown(show_project_structure())
            
        with gr.Tab("Configuration"):
            gr.Markdown("## Default Configuration")
            gr.Markdown("""
            - **Agent Type**: custom
            - **Max Steps**: 100
            - **Max Actions Per Step**: 10
            - **Use Vision**: True
            - **Tool Calling Method**: auto
            - **LLM Provider**: openai
            - **LLM Model Name**: gpt-4o
            - **LLM Context Length**: 32000
            - **LLM Temperature**: 1.0
            """)

# Run the app
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7789)
