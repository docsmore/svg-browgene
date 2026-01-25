# BrowGene v2 Docker Deployment

This directory contains the BrowGene v2 service that provides a local API server for browser automation using AI agents.

## Features

- **Local API Server**: FastAPI-based server running on port 7793
- **Browser Automation**: Uses browser-use library v0.1.35 with Playwright
- **AI Integration**: Supports Google Gemini, OpenAI, Anthropic, and other LLM providers
- **Docker Ready**: Fully containerized with health checks and proper networking

## Quick Start

### Using Docker Compose

1. **Start the service**:
   ```bash
   cd /Users/sovenshrivastav/svg-projects/svg-docker
   docker-compose up svg-browgene-v2
   ```

2. **Access the API**:
   - API Server: http://localhost:7793
   - API Documentation: http://localhost:7793/docs
   - Health Check: http://localhost:7793/health

### Environment Variables

Required environment variables (set in `.env.local`):

```bash
# LLM API Keys (at least one required)
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Optional: Other providers
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_KEY=your_azure_key
DEEPSEEK_API_KEY=your_deepseek_key
```

## API Usage

### Run a Task

```bash
curl -X POST "http://localhost:7793/api/browgene/run_task" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Go to https://example.com and extract the main heading",
    "agent_type": "org",
    "llm_provider": "google",
    "llm_model_name": "gemini-2.0-flash-exp",
    "max_steps": 10,
    "headless": true,
    "use_vision": true
  }'
```

### Check Task Status

```bash
curl "http://localhost:7793/api/browgene/task_status/{task_id}"
```

## Integration with PulseGene

The BrowGene v2 node in PulseGene workflows can connect to this Docker service by setting:

```
api_endpoint: http://svg-browgene-v2:7793
```

Or from outside Docker:

```
api_endpoint: http://localhost:7793
```

## Development

To run in development mode:

```bash
cd /Users/sovenshrivastav/svg-projects/svg-browgene
source .venv/bin/activate
python -m uvicorn api_server:app --host 0.0.0.0 --port 7793 --reload
```

## Troubleshooting

1. **Check container logs**:
   ```bash
   docker logs svg-browgene-v2
   ```

2. **Verify health**:
   ```bash
   curl http://localhost:7793/health
   ```

3. **Test API docs**:
   Visit http://localhost:7793/docs for interactive API documentation.
