# BrowGene v2: R&D Analysis — Building Beyond browser-use

**Date:** February 27, 2025  
**Author:** Solvrays Engineering  
**Status:** R&D Phase

---

## 1. Executive Summary

**browser-use** is a popular open-source Python library (MIT License, 60k+ GitHub stars) that makes websites accessible to AI agents. It uses Playwright under the hood and LLMs to autonomously navigate, interact with, and extract data from web pages.

**Key finding:** The open-source version is a solid foundation (~70-78% task completion on benchmarks) but has significant gaps compared to the Cloud/Pro version, and both have fundamental limitations that create an opportunity for us to build something substantially better — especially for **enterprise insurance workflows**.

---

## 2. browser-use Open Source — What You Get

### Architecture
```
Task (natural language) → LLM Agent Loop → Playwright Actions → Results
                            ↑                    ↓
                     Page State (DOM + Screenshot) ← Browser
```

### Core Features (Open Source)
| Feature | Status | Notes |
|---------|--------|-------|
| Autonomous agent loop | ✅ | LLM receives page state, decides next action, executes, loops |
| Multi-LLM support | ✅ | OpenAI, Anthropic, Google, Ollama, DeepSeek, Groq, etc. |
| Vision mode | ✅ | Screenshots + DOM, auto/on/off modes |
| Multi-tab browsing | ✅ | Agent can open/switch tabs |
| Custom tools | ✅ | Add custom functions the agent can call |
| Initial actions | ✅ | Pre-programmed actions before AI takes over |
| Structured output | ✅ | Pydantic model validation for results |
| Sensitive data handling | ✅ | Dictionary-based masking |
| Fallback LLM | ✅ | Backup model on primary failure |
| Flash mode | ✅ | Fast mode skipping evaluation/thinking |
| GIF recording | ✅ | Visual playback of agent actions |
| CLI interface | ✅ | Command-line browser control |
| Conversation saving | ✅ | Save full conversation history |
| Custom system prompts | ✅ | Override or extend system messages |
| Real browser profiles | ✅ | Reuse existing Chrome profiles with saved logins |
| Code Agent | ✅ | Alternative agent that generates Python code |

### What Open Source Does NOT Have
| Feature | Cloud/Pro Only | Impact |
|---------|---------------|--------|
| Stealth browsers | ✅ Pro only | Anti-detection, fingerprinting bypass |
| CAPTCHA solving | ✅ Pro only | Critical for real-world automation |
| Proxy rotation | ✅ Pro only | Geo-targeting, IP rotation |
| Skills system | ✅ Pro only | Turn websites into deterministic API endpoints |
| Skills marketplace | ✅ Pro only | Share/clone reusable automation skills |
| ChatBrowserUse LLM | ✅ Pro only | Custom model optimized for browser tasks (15x cheaper, 6x faster) |
| Sandbox deployment | ✅ Pro only | `@sandbox()` decorator for cloud execution |
| Cookie sync to cloud | ✅ Pro only | Profile persistence across cloud sessions |
| Scalable infra | ✅ Pro only | Auto-scaling, memory management |
| Browser session API | ✅ Pro only | Direct CDP access to cloud browsers |

---

## 3. Benchmarks & Known Limitations

### Performance (WebVoyager benchmark)
| Configuration | Task Completion |
|---------------|----------------|
| browser-use + GPT-4.1 Vision | ~72% |
| browser-use + Claude Opus | ~78% |
| Stagehand + Claude Sonnet | ~75% |
| Hand-written Playwright scripts | ~98% |

### Cost Per Task (GPT-4.1 pricing)
| Complexity | Cost |
|------------|------|
| Simple (5 steps) | $0.02 - $0.08 |
| Complex (20 steps) | $0.08 - $0.30 |
| ChatBrowserUse simple | ~$0.03 (~33 tasks/dollar) |

### Known Limitations of browser-use (Open Source)
1. **Reliability ceiling ~78%** — Good for prototyping, not production-grade for critical workflows
2. **No deterministic fallback** — Pure AI approach, no hybrid AI+scripted path
3. **Single browser context** — No true parallel multi-browser orchestration
4. **No workflow persistence** — No checkpoint/resume on failure
5. **No session recording/playback** — Can't record and replay interactions
6. **No enterprise auth** — No SSO, no vault integration, no credential management
7. **No audit trail** — No compliance logging of what the agent did
8. **Expensive at scale** — Each step = LLM call, no caching or optimization
9. **No scheduling** — No cron-like scheduled execution
10. **No multi-agent coordination** — Agents can't collaborate on complex tasks
11. **No domain-specific optimization** — Generic prompts, no insurance/financial domain knowledge
12. **No human-in-the-loop** — No approval workflows, no intervention points
13. **Memory is conversation-only** — No long-term knowledge base across tasks
14. **No PDF/document interaction** — Can't handle file uploads/downloads intelligently
15. **No iframe/shadow DOM specialization** — Struggles with complex embedded content

---

## 4. Competitor Landscape

### Stagehand (by Browserbase)
- **Approach:** Surgical AI primitives (`act`, `extract`, `observe`) on top of Playwright
- **Strengths:** TypeScript-native, Zod schemas for typed extraction, hybrid deterministic+AI
- **Weakness:** Less autonomous, requires more developer guidance
- **Key insight:** Their hybrid approach (AI for ambiguous parts, Playwright for deterministic parts) is the winning architecture

### Skyvern
- **Approach:** Cloud-first, visual grounding with bounding boxes
- **Strengths:** Good at form filling, visual element identification
- **Weakness:** Closed source cloud dependency, expensive

### Anthropic Computer Use / OpenAI Operator
- **Approach:** Foundation model with native computer interaction
- **Strengths:** Can interact with any application (not just browsers)
- **Weakness:** Very expensive per action, slow, limited control

### Our Current BrowGene
- **What we have:** Fork of browser-use web-ui with CustomAgent, CustomBrowser, Gradio WebUI, API server with script mode + AI mode
- **Strengths:** Already integrated with Solvrays ecosystem, Playwright script mode, VNC viewing
- **Gaps:** Same limitations as upstream browser-use, no enterprise features

---

## 5. Where browser-use Falls Short for Our Use Cases

### Insurance-Specific Gaps

| Use Case | What's Needed | browser-use Gap |
|----------|--------------|-----------------|
| **Legacy PAS data entry** | Navigate complex multi-frame insurance portals, fill forms across 10+ screens | No checkpoint/resume, no form memory, ~72% reliability insufficient |
| **Claims intake automation** | Upload documents, fill structured forms, handle CAPTCHA | No file handling, no CAPTCHA, no stealth |
| **Carrier portal scraping** | Log into 50+ carrier portals, extract policy/billing data | No credential vault, no parallel sessions, no scheduling |
| **Compliance auditing** | Record every action taken, produce audit trail | Zero audit capability |
| **Rate comparison** | Navigate multiple carrier sites simultaneously, extract quotes | No multi-agent coordination, no structured extraction guarantee |
| **Policy issuance workflow** | 20+ step workflow across multiple systems with approvals | No human-in-the-loop, no workflow persistence, no retry |
| **Renewal processing** | Batch process hundreds of renewals across carrier portals | No batch orchestration, no queue management |
| **Document download/upload** | Download dec pages, upload to ClouGene, process with DocuGene | No file pipeline integration |

---

## 6. BrowGene v2 — Architecture Proposal

### Vision
Build an **enterprise-grade autonomous browser agent platform** that combines the best of browser-use's AI agent loop with Stagehand's hybrid approach, plus enterprise features no one else has.

### Core Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      BrowGene v2 Platform                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │  Task Queue  │  │  Scheduler   │  │  Human-in-the-Loop UI   │ │
│  │  (Redis/BQ)  │  │  (Cron/Event)│  │  (Approval workflows)   │ │
│  └──────┬──────┘  └──────┬───────┘  └───────────┬─────────────┘ │
│         │                │                       │                 │
│  ┌──────▼────────────────▼───────────────────────▼─────────────┐ │
│  │                   Orchestrator Layer                          │ │
│  │  - Multi-agent coordination                                  │ │
│  │  - Workflow engine (PulseGene integration)                   │ │
│  │  - Checkpoint/Resume system                                  │ │
│  │  - Batch processing manager                                  │ │
│  └──────────────────────┬──────────────────────────────────────┘ │
│                         │                                         │
│  ┌──────────────────────▼──────────────────────────────────────┐ │
│  │                    Agent Layer                                │ │
│  │                                                               │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │ │
│  │  │  AI Agent    │  │ Script Agent  │  │  Hybrid Agent     │  │ │
│  │  │ (browser-use │  │ (Playwright   │  │ (AI + Script      │  │ │
│  │  │  agent loop) │  │  deterministic│  │  combined)        │  │ │
│  │  └──────┬──────┘  └──────┬───────┘  └────────┬──────────┘  │ │
│  │         │                │                    │              │ │
│  │  ┌──────▼────────────────▼────────────────────▼──────────┐  │ │
│  │  │              Unified Action Engine                      │  │ │
│  │  │  - Click, Type, Scroll, Navigate, Extract, Upload      │  │ │
│  │  │  - Smart element detection (DOM + Vision + Heuristics) │  │ │
│  │  │  - Action caching & optimization                       │  │ │
│  │  └──────────────────────┬────────────────────────────────┘  │ │
│  └──────────────────────────┬──────────────────────────────────┘ │
│                              │                                    │
│  ┌──────────────────────────▼──────────────────────────────────┐ │
│  │                   Browser Layer                               │ │
│  │  - Playwright (primary)                                       │ │
│  │  - Multi-browser pool (parallel sessions)                     │ │
│  │  - Stealth/anti-detection (our own)                          │ │
│  │  - Proxy management                                           │ │
│  │  - Session persistence & cookie management                    │ │
│  │  - File upload/download pipeline                              │ │
│  └──────────────────────────┬──────────────────────────────────┘ │
│                              │                                    │
│  ┌──────────────────────────▼──────────────────────────────────┐ │
│  │                Enterprise Services Layer                      │ │
│  │  - Credential Vault (encrypted, per-carrier)                  │ │
│  │  - Audit Trail (every action logged with screenshots)         │ │
│  │  - Skills Registry (reusable automation recipes)              │ │
│  │  - Knowledge Base (domain-specific, per-carrier portal)       │ │
│  │  - Recording/Playback (RPA-style)                             │ │
│  │  - Metrics & Cost Tracking                                    │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │              Solvrays Gene Integration                         │ │
│  │  PulseGene │ RouteGene │ ClouGene │ DocuGene │ ScribeGene    │ │
│  └──────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## 7. Key Innovations Over browser-use

### 7.1 Hybrid Agent Mode (AI + Deterministic)
The **#1 innovation**. Instead of pure AI (unreliable) or pure scripted (brittle):

```python
# Define a hybrid workflow
workflow = BrowGeneWorkflow(
    name="erie_policy_entry",
    steps=[
        # Deterministic: Navigate to known URL, login with vault creds
        ScriptStep("navigate", url="https://portal.erieinsurance.com"),
        ScriptStep("login", credential_key="erie_portal"),
        
        # AI: Handle dynamic dashboard, find the right menu
        AIStep("Navigate to new policy entry form"),
        
        # Deterministic: Fill known fields from structured data
        ScriptStep("fill_form", mapping={
            "#insured_name": "${payload.insured.name}",
            "#policy_effective": "${payload.effective_date}",
        }),
        
        # AI: Handle dynamic fields, dropdowns with search
        AIStep("Select the coverage type '${payload.coverage_type}' from the dropdown"),
        
        # Human checkpoint: Review before submission
        HumanCheckpoint("Review the form before submitting", timeout_minutes=30),
        
        # Deterministic: Submit
        ScriptStep("click", selector="#submit_button"),
    ]
)
```

**Why this wins:** Scripted steps are fast and free (no LLM cost). AI steps handle the unpredictable parts. Human checkpoints prevent costly mistakes.

### 7.2 Skills Registry (Our Own, No Cloud Dependency)

```python
# Create a reusable skill from a successful run
skill = browgene.create_skill(
    name="erie_get_policy_status",
    description="Check policy status on Erie portal",
    parameters={"policy_number": "string"},
    recording_id="rec_abc123",  # Learned from a recorded session
    fallback="ai",  # If script fails, use AI
)

# Execute it like an API
result = await browgene.execute_skill(
    "erie_get_policy_status",
    parameters={"policy_number": "POL-12345"}
)
```

### 7.3 Checkpoint/Resume System

```python
# Workflow automatically saves state after each step
# If step 15 of 20 fails:
execution = await browgene.resume(
    execution_id="exec_abc123",
    from_step=15,  # Resume from failure point
    variable_overrides={"retry_selector": "#alt_submit_btn"}
)
```

### 7.4 Multi-Agent Coordination

```python
# Fan out: Scrape 50 carrier portals in parallel
results = await browgene.parallel_execute(
    skill="get_renewal_quote",
    targets=[
        {"carrier": "erie", "policy": "POL-001"},
        {"carrier": "travelers", "policy": "POL-002"},
        {"carrier": "hanover", "policy": "POL-003"},
        # ... 47 more
    ],
    max_concurrent=10,
    on_failure="continue",  # Don't stop batch for one failure
)
```

### 7.5 Enterprise Audit Trail

```python
# Every action automatically logged
{
    "execution_id": "exec_abc123",
    "step": 5,
    "action": "click",
    "selector": "#submit_policy",
    "screenshot_before": "s3://audit/exec_abc123/step5_before.png",
    "screenshot_after": "s3://audit/exec_abc123/step5_after.png",
    "timestamp": "2025-02-27T23:10:00Z",
    "user": "system@solvrays.ai",
    "duration_ms": 1200,
    "llm_tokens_used": 0,  # This was a script step
    "cost": 0.00
}
```

### 7.6 Carrier Portal Knowledge Base

```python
# Per-carrier portal knowledge stored in Qdrant/ScribeGene
knowledge = {
    "carrier": "erie_insurance",
    "portal_url": "https://portal.erieinsurance.com",
    "login_type": "username_password",
    "mfa_type": "email_code",
    "known_selectors": {
        "policy_search": "#policySearch",
        "new_quote": "a[href='/quotes/new']",
    },
    "quirks": [
        "Session expires after 15 minutes of inactivity",
        "Download buttons use JavaScript popups, need to handle them",
        "Rate page loads slowly, need 10s timeout",
    ],
    "last_updated": "2025-02-20",
    "reliability_score": 0.95,
}
```

### 7.7 Recording & Playback (RPA-Style)

A major feature browser-use completely lacks. Users can:
1. **Record** a manual browser session (clicks, typing, navigation)
2. **Annotate** which parts are dynamic (AI handles these)
3. **Export** as a reusable skill/workflow
4. **Playback** with different data
5. **Auto-heal** when the site changes (AI fills the gap)

---

## 8. Tech Stack Recommendation

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Agent core** | Python 3.11+ | browser-use is Python, LangChain ecosystem |
| **Browser engine** | Playwright | Industry standard, multi-browser support |
| **LLM layer** | LangChain + direct APIs | Flexibility, multi-provider |
| **Stealth** | playwright-stealth + custom patches | No cloud dependency |
| **Task queue** | Redis + BullMQ (or Celery) | Proven, scalable |
| **Scheduler** | APScheduler or Celery Beat | Cron + event-driven |
| **Knowledge base** | Qdrant (ScribeGene) | Already in our stack |
| **Credential vault** | Encrypted Prisma model or HashiCorp Vault | Enterprise-grade |
| **Audit storage** | PostgreSQL + S3 (screenshots) | Queryable + blob storage |
| **API** | FastAPI | Already using for BrowGene |
| **UI** | Next.js (Solvrays) | Integration with existing platform |
| **Orchestration** | PulseGene integration | Workflow engine already built |

---

## 9. Build vs Buy Analysis

### Option A: Fork & Enhance browser-use (Recommended)
- **Effort:** 3-4 months for v1
- **Pros:** MIT license, solid foundation, active community, Python ecosystem
- **Cons:** Must maintain our fork, upstream breaking changes
- **Strategy:** Fork the core library, keep our `CustomAgent` pattern, add enterprise layers on top

### Option B: Build from Scratch on Playwright
- **Effort:** 6-8 months for v1
- **Pros:** Full control, no upstream dependency
- **Cons:** Massive effort to recreate what browser-use already does well

### Option C: Use browser-use as dependency (no fork)
- **Effort:** 2-3 months for v1
- **Pros:** Automatic upstream updates
- **Cons:** Limited customization of core agent loop

### Recommendation: **Option A** — Fork & Enhance
We already have a fork (svg-browgene). The core agent loop works. We need to:
1. Update to latest browser-use (v0.9.6+)
2. Add enterprise layers on top
3. Build the hybrid agent mode
4. Integrate with Solvrays Gene ecosystem

---

## 10. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)
- [ ] Update svg-browgene to latest browser-use
- [ ] Implement Hybrid Agent Mode (Script + AI + Human steps)
- [ ] Build Credential Vault (encrypted storage per carrier)
- [ ] Add Audit Trail system (action logging + screenshots)
- [ ] Create FastAPI v2 endpoints for new features

### Phase 2: Enterprise Features (Weeks 5-8)
- [ ] Skills Registry (create, execute, manage reusable automations)
- [ ] Checkpoint/Resume system (save state, resume from failure)
- [ ] Recording/Playback (capture manual sessions, replay with AI)
- [ ] Multi-browser pool (parallel session management)
- [ ] Scheduling system (cron-based and event-driven)

### Phase 3: Intelligence (Weeks 9-12)
- [ ] Carrier Portal Knowledge Base (Qdrant integration)
- [ ] Domain-specific prompts (insurance vocabulary, form patterns)
- [ ] Smart element detection (DOM + Vision + Heuristics + History)
- [ ] Action caching (skip LLM for known stable pages)
- [ ] Cost optimization (route to cheapest model per step complexity)

### Phase 4: Integration & Scale (Weeks 13-16)
- [ ] PulseGene integration (BrowGene as workflow node)
- [ ] ClouGene integration (file upload/download pipeline)
- [ ] DocuGene integration (document processing from browser)
- [ ] RouteGene integration (task assignment to browser agents)
- [ ] Multi-agent coordination (fan-out parallel execution)
- [ ] Solvrays UI components (monitoring, configuration, execution viewer)

---

## 11. Open Source Version Assessment — Final Verdict

### Is the open-source version as good as Pro?

**No.** The Pro/Cloud version adds critical capabilities:

| Capability | Impact | Can We Build It? |
|-----------|--------|-------------------|
| Stealth browsers | High — many sites block automation | ✅ Yes, playwright-stealth + custom |
| CAPTCHA solving | High — blocks many workflows | ⚠️ Partial — can integrate 2captcha/anti-captcha |
| Proxy rotation | Medium — needed for scale | ✅ Yes, easy to implement |
| Skills system | High — key productivity feature | ✅ Yes, and we can make it better |
| Custom LLM | Medium — cost optimization | ✅ We can fine-tune our own |
| Cloud scale | Medium — needed for production | ✅ Docker + K8s handles this |

### Can we make it better than Pro?

**Yes.** Browser-use Pro is a generic tool. We can build:
1. **Insurance-domain expertise** baked into prompts and knowledge base
2. **Hybrid agent mode** that Pro doesn't have
3. **Enterprise compliance** features (audit, approvals, credential vault)
4. **Deep Gene ecosystem integration** that no competitor can match
5. **Recording/Playback** that lowers the bar for non-technical users
6. **Checkpoint/Resume** for mission-critical long-running workflows
7. **Multi-agent coordination** for batch carrier operations

### The Moat
Browser-use is a **horizontal tool** for general web automation. We're building a **vertical tool** optimized for insurance operations with:
- Pre-built carrier portal knowledge
- Insurance-specific form understanding
- Compliance-grade audit trails
- Integration with our existing document, workflow, and storage Genes

This vertical specialization is what makes our version substantially more valuable than both the open-source and Pro versions for our target market.

---

## 12. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| browser-use upstream breaking changes | High | Medium | Pin version, selective merge |
| LLM costs at scale | Medium | High | Hybrid mode, action caching, cheap models for simple steps |
| Anti-bot detection escalation | High | High | Stealth patches, proxy rotation, real browser profiles |
| Carrier portal changes | High | Medium | Knowledge base updates, AI auto-healing, monitoring |
| Reliability < 90% | Medium | High | Hybrid mode, human checkpoints, retry logic |

---

## 13. Next Steps

1. **Align on scope** — Which Phase 1 features are highest priority?
2. **Update fork** — Bring svg-browgene to latest browser-use version
3. **Prototype hybrid mode** — Build proof of concept with one carrier portal
4. **Define carrier targets** — Which 5 carrier portals do we automate first?
5. **Set up infrastructure** — Redis, Qdrant collection, audit tables

---

*This document will be updated as the R&D progresses.*
