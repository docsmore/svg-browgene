# BrowGene v2 — Technical Design

## Three-Mode Architecture: Explore → Learn → Execute

```
┌────────────────────────────────────────────────────────────────────┐
│                        BrowGene v2                                  │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐ │
│  │  EXPLORE      │───▶│  LEARN        │───▶│  EXECUTE              │ │
│  │  (browser-use │    │  (Record &    │    │  (Deterministic       │ │
│  │   AI agent)   │    │   Convert)    │    │   step replay)        │ │
│  └──────┬───────┘    └──────┬───────┘    └──────────┬───────────┘ │
│         │                   │                        │              │
│  ┌──────▼───────────────────▼────────────────────────▼───────────┐ │
│  │                 Playwright Transport Layer                      │ │
│  │  click | right_click | fill | scroll | extract | navigate      │ │
│  │  wait_for | screenshot | evaluate_js | upload | download       │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                 Task Framework (from DeskGene)                  │ │
│  │  BrowserTask | BrowserStep | TaskManager | TaskRecorder        │ │
│  └────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

---

## Mode 1: EXPLORE (browser-use AI Agent)

**When:** You don't know the page structure, need to discover workflows.

Uses browser-use's AI agent loop to navigate unknown pages. The agent takes screenshots, analyzes DOM, and decides actions autonomously. Every action the agent takes is **recorded** as a potential step for later conversion.

```python
result = await browgene.explore(
    task="Find the policy search page and search for policy POL-12345",
    llm=ChatOpenAI(model="gpt-4o"),
    record=True,  # Record actions for Learn mode
)
# result.recorded_steps → list of actions the AI took
# result.final_state → page URL, DOM snapshot, screenshot
```

## Mode 2: LEARN (Record & Convert)

**When:** You've explored a workflow and want to make it repeatable.

Takes the recorded AI actions from Explore mode (or manual recording) and converts them into deterministic BrowserStep sequences. User can edit, parameterize, and test the steps.

```python
# From an explore session
task = browgene.learn_from_exploration(
    exploration_id="exp_abc123",
    task_name="search_policy",
    parameterize={
        "step_3.value": "${policy_number}",  # Make the search input a variable
    }
)

# Or record manually
async with browgene.record("manual_task") as recorder:
    # User performs actions in the browser
    # Each action is captured as a BrowserStep
    pass
```

## Mode 3: EXECUTE (Deterministic Replay)

**When:** You have a known workflow defined as steps. Fast, free (no LLM cost), reliable.

Executes BrowserStep sequences directly via Playwright. No AI involved unless a step explicitly requests AI fallback.

```python
result = await browgene.execute(
    task_name="search_policy",
    parameters={"policy_number": "POL-12345"},
    on_step_failure="ai_fallback",  # Use AI if a step fails
)
```

---

## Step Types

Each step maps directly to a Playwright action:

| Step Type | Playwright Method | Parameters | Example |
|-----------|------------------|------------|---------|
| `navigate` | `page.goto(url)` | `url` | Go to a URL |
| `click` | `page.click(selector)` | `selector`, `button` (left/right) | Click an element |
| `right_click` | `page.click(selector, button='right')` | `selector` | Right-click for context menu |
| `fill` | `page.fill(selector, value)` | `selector`, `value` | Type into an input field |
| `type` | `page.type(selector, text)` | `selector`, `text`, `delay` | Type character by character |
| `select` | `page.select_option(selector, value)` | `selector`, `value` | Select dropdown option |
| `scroll` | `page.mouse.wheel(dx, dy)` | `dx`, `dy` or `selector` | Scroll page or element |
| `wait_for` | `page.wait_for_selector(selector)` | `selector`, `state`, `timeout` | Wait for element |
| `wait` | `asyncio.sleep(seconds)` | `seconds` | Fixed delay |
| `screenshot` | `page.screenshot()` | `path`, `full_page` | Capture screenshot |
| `extract` | `page.evaluate(js)` | `js_expression`, `output_schema` | Extract data from page |
| `keyboard` | `page.keyboard.press(key)` | `key` | Press keyboard key (Tab, Enter, etc.) |
| `upload` | `page.set_input_files(selector, files)` | `selector`, `file_paths` | Upload file |
| `evaluate` | `page.evaluate(js)` | `js_expression` | Run arbitrary JS |
| `ai_analyze` | Gemini Vision | `prompt`, `output_schema` | AI analyzes current page |
| `ai_act` | browser-use agent step | `instruction` | AI performs one action |
| `hold_data` | internal memory | `key`, `extract_js` | Store extracted data in memory |
| `merge_data` | internal memory | `sources`, `merge_strategy` | Merge stored data |
| `assert` | conditional check | `condition`, `on_fail` | Verify page state |

---

## Data Model

### BrowserStep

```python
@dataclass
class BrowserStep:
    step_type: str              # One of the step types above
    params: Dict[str, Any]      # Parameters for the step
    description: str            # Human-readable description
    delay_ms: int = 500         # Delay after step
    take_screenshot: bool = True
    on_failure: str = "stop"    # "stop" | "skip" | "ai_fallback" | "retry"
    max_retries: int = 1
    timeout_ms: int = 30000
    # Runtime fields
    status: str = "pending"     # "pending" | "running" | "success" | "failed" | "skipped"
    screenshot_before: Optional[str] = None
    screenshot_after: Optional[str] = None
    extracted_data: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None
```

### BrowserTask

```python
@dataclass
class BrowserTask:
    name: str
    description: str
    steps: List[BrowserStep]
    parameters: Dict[str, str]    # Variable definitions
    created_at: str
    updated_at: str
    source: str                   # "manual" | "explored" | "recorded"
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    # Execution context
    start_url: Optional[str] = None
    requires_auth: bool = False
    auth_credential_key: Optional[str] = None
```

### TaskExecution

```python
@dataclass
class TaskExecution:
    execution_id: str
    task_name: str
    parameters: Dict[str, Any]      # Resolved parameter values
    mode: str                        # "explore" | "execute" | "hybrid"
    status: str                      # "running" | "completed" | "failed"
    steps_completed: int
    steps_total: int
    start_time: str
    end_time: Optional[str]
    memory: Dict[str, Any]           # hold_data storage
    screenshots: List[str]
    extracted_data: List[Any]
    error: Optional[str]
```

---

## API Endpoints

### Task Management
```
POST   /api/tasks                    - Create a new task
GET    /api/tasks                    - List all tasks
GET    /api/tasks/{name}             - Get task details
PUT    /api/tasks/{name}             - Update a task
DELETE /api/tasks/{name}             - Delete a task
```

### Execution
```
POST   /api/execute                  - Execute a task (Execute mode)
POST   /api/explore                  - Start an exploration (Explore mode)
POST   /api/execute/{id}/resume      - Resume a failed execution
GET    /api/executions               - List executions
GET    /api/executions/{id}          - Get execution status & results
GET    /api/executions/{id}/screenshots - Get execution screenshots
```

### Learning
```
POST   /api/learn/from-exploration   - Convert exploration to task
POST   /api/learn/record/start       - Start manual recording
POST   /api/learn/record/stop        - Stop recording, save as task
```

### Browser
```
POST   /api/browser/launch           - Launch a browser session
POST   /api/browser/close            - Close browser session
GET    /api/browser/status            - Get browser status
POST   /api/browser/screenshot        - Take a screenshot
```

### Memory (hold/merge data)
```
GET    /api/memory/{execution_id}     - Get stored data for execution
POST   /api/memory/{execution_id}     - Store data during execution
```

---

## Workflow Example: Legacy PAS Data Entry

```json
{
  "name": "erie_policy_search",
  "description": "Search for a policy on Erie Insurance portal",
  "start_url": "https://portal.erieinsurance.com",
  "requires_auth": true,
  "auth_credential_key": "erie_portal",
  "parameters": {
    "policy_number": "string"
  },
  "steps": [
    {
      "step_type": "wait_for",
      "params": {"selector": "#policySearch", "timeout": 10000},
      "description": "Wait for policy search field to appear"
    },
    {
      "step_type": "fill",
      "params": {"selector": "#policySearch", "value": "${policy_number}"},
      "description": "Enter policy number"
    },
    {
      "step_type": "click",
      "params": {"selector": "#searchBtn"},
      "description": "Click search button"
    },
    {
      "step_type": "wait_for",
      "params": {"selector": ".policy-result", "timeout": 15000},
      "description": "Wait for search results"
    },
    {
      "step_type": "right_click",
      "params": {"selector": ".policy-result:first-child"},
      "description": "Right-click first result for context menu"
    },
    {
      "step_type": "click",
      "params": {"selector": ".context-menu .view-details"},
      "description": "Click 'View Details' in context menu"
    },
    {
      "step_type": "wait_for",
      "params": {"selector": ".policy-details", "timeout": 10000},
      "description": "Wait for policy details to load"
    },
    {
      "step_type": "extract",
      "params": {
        "js_expression": "JSON.stringify({status: document.querySelector('.policy-status').textContent, premium: document.querySelector('.premium-amount').textContent, effective: document.querySelector('.effective-date').textContent})"
      },
      "description": "Extract policy details as JSON"
    },
    {
      "step_type": "hold_data",
      "params": {"key": "policy_details", "from_step": 7},
      "description": "Store extracted policy data in memory"
    }
  ]
}
```

---

## File Structure

```
svg-browgene/
├── browgene/                    # Core library
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── task.py              # BrowserTask, BrowserStep
│   │   ├── task_manager.py      # TaskManager (save/load/list)
│   │   ├── executor.py          # Execute mode - step runner
│   │   ├── explorer.py          # Explore mode - browser-use wrapper
│   │   ├── learner.py           # Learn mode - convert exploration to task
│   │   └── recorder.py          # Manual recording
│   ├── transport/
│   │   ├── __init__.py
│   │   ├── playwright_driver.py # Playwright wrapper with all actions
│   │   └── browser_pool.py      # Multi-browser session management
│   ├── memory/
│   │   ├── __init__.py
│   │   └── execution_memory.py  # hold_data / merge_data support
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── vision.py            # Screenshot analysis (Gemini/GPT-4V)
│   │   └── fallback.py          # AI fallback when steps fail
│   └── utils/
│       ├── __init__.py
│       ├── screenshots.py       # Screenshot management
│       └── logging.py           # Structured logging
├── api/
│   ├── __init__.py
│   └── server.py                # FastAPI server
├── tasks/                       # Saved task definitions (JSON)
├── executions/                  # Execution logs and screenshots
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

1. **Playwright, not Selenium** — Playwright has native right-click, better async, auto-wait, and is what browser-use already uses.

2. **browser-use as optional dependency** — Explore mode imports browser-use. Execute mode does not need it at all. This keeps the core lightweight.

3. **JSON task files** — Same as DeskGene. Tasks are JSON files that can be version controlled, shared, and edited manually.

4. **Parameter interpolation** — `${variable_name}` syntax in step params, resolved at execution time. Allows reusable tasks with different inputs.

5. **AI fallback per step** — Each step can define `on_failure: "ai_fallback"` to let the AI agent try when deterministic execution fails. This is the hybrid magic.

6. **Memory system** — `hold_data` and `merge_data` steps let you accumulate data across steps (e.g., extract from page 1, filter on page 2, merge results).

7. **Screenshot at every step** — Optional but default on. Creates a visual audit trail like DeskGene.
