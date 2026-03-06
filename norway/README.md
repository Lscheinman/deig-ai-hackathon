# JONE Agent - Norwegian HR Graph Agent

**JOb Norsk Engine** - A deterministic HR graph assistant for military force readiness and personnel capability analysis.

## Overview

JONE is an AI-powered agent that queries Norwegian military HR data stored in SAP HANA Graph. It provides:

- **Skills & Qualifications Search**: Find personnel skills, qualifications, and competencies
- **Task-Skill Mapping**: Discover which tasks require which skills
- **Force Element Personnel**: Query personnel assigned to military units
- **Readiness Analysis**: Assess unit readiness, training gaps, and deployment availability
- **Job Announcements**: Generate job postings based on task requirements

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      JONE Agent                              │
├─────────────────────────────────────────────────────────────┤
│  FastAPI Server (run.py)                                     │
│    └── /jone/agent/jone/chat    - Chat endpoint             │
│    └── /jone/agent/jone/stream  - SSE streaming             │
│    └── /jone/agent/jone/test    - Direct op testing         │
├─────────────────────────────────────────────────────────────┤
│  Agent Layer (src/agent_jone/)                              │
│    └── api.py         - FastAPI routes & LLM orchestration  │
│    └── engine.py      - Deterministic HANA Graph operations │
│    └── prompts.py     - LLM system prompts                  │
│    └── config.py      - Pydantic models & config loaders    │
│    └── plan_validation.py - Plan normalization              │
├─────────────────────────────────────────────────────────────┤
│  Commons Layer (commons/)                                    │
│    └── ai_core/       - SAP AI Core / GenAI Hub client      │
│    └── hana/          - SAP HANA connection utilities       │
│    └── core/          - Configuration & settings            │
├─────────────────────────────────────────────────────────────┤
│  External Services                                           │
│    └── SAP AI Core    - LLM (GPT-4o)                        │
│    └── SAP HANA Cloud - Graph database (DFS_HR schema)      │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Prerequisites

- Python 3.10+
- SAP HANA Cloud instance with HR graph data
- SAP AI Core deployment (optional - falls back to heuristic mode)

### 2. Setup

```bash
# Clone or copy this directory
cd norway

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and edit environment file
cp .env.example .env
# Edit .env with your credentials
```

### 3. Run

```bash
# Using the runner script
python run.py

# Or with uvicorn directly
uvicorn run:app --host 0.0.0.0 --port 8080 --reload
```

### 4. Test

Visit http://localhost:8080/docs for the Swagger UI.

**Example chat request:**
```bash
curl -X POST http://localhost:8080/jone/agent/jone/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Who has project management skills?"}'
```

## Available Operations

### HR Graph Operations (Norwegian Skills/Tasks/Qualities)

| Operation | Description |
|-----------|-------------|
| `search_nodes` | Search nodes by text (skills, tasks, qualities) |
| `tasks_for_skill` | Get tasks linked to a qualification |
| `qualities_for_task` | Get personal qualities for a task |
| `neighbors_subgraph` | Graph neighborhood traversal |
| `shortest_path` | Find path between two nodes |
| `nl_search` | Natural language search with keywords |
| `resolve_and_expand` | Two-step: search + expand to related items |

### Force Element Personnel Operations

| Operation | Description |
|-----------|-------------|
| `personnel_for_fe` | Get personnel assigned to a unit |
| `fe_personnel_by_competency` | Find personnel with specific competency |
| `fe_readiness` | Get unit readiness metrics |
| `search_fe_personnel` | Search personnel by name/role/grade |
| `position_status` | Get position fill status for a unit |

### Military HR Operations

| Operation | Description |
|-----------|-------------|
| `personnel_training` | Training records for personnel |
| `personnel_qualifications` | Weapon/vehicle/system qualifications |
| `personnel_deployments` | Deployment history |
| `personnel_medical` | Medical readiness records |
| `training_gaps` | Find expired/missing training |
| `qualification_status` | Qualification expiry status |
| `deployment_availability` | Who is available for deployment |

## Configuration

### Environment Variables

See `.env.example` for all configuration options.

Key variables:
- `HANA_HOST`, `HANA_PORT`, `HANA_USER`, `HANA_PASSWORD` - HANA connection
- `AICORE_*` - SAP AI Core credentials
- `JONE_TARGET_SCHEMA` - Default: `DFS_HR`

### Agent Rules

Edit `src/agent_jone/rules.yaml` to customize:
- Default/max limits
- Confidence thresholds
- Allowed operations

### Task Routing

Edit `src/agent_jone/task_routing.json` to:
- Add/modify operations
- Define parameter schemas
- Add examples

## Project Structure

```
norway/
├── run.py                    # FastAPI entry point
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
├── README.md                 # This file
├── commons/                  # Shared utilities
│   ├── ai_core/              # LLM client
│   ├── core/                 # Configuration
│   └── hana/                 # HANA connections
└── src/
    └── agent_jone/           # JONE agent source
        ├── api.py            # FastAPI routes
        ├── engine.py         # Graph operations
        ├── config.py         # Models & config
        ├── prompts.py        # LLM prompts
        ├── plan_validation.py
        ├── rules.yaml        # Agent rules
        ├── task_routing.json # Operation definitions
        └── utils/
            └── sportsone_urls.py  # Drill-down URLs
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Agent info |
| `/health` | GET | Health check |
| `/jone/agent/jone/chat` | POST | Chat endpoint |
| `/jone/agent/jone/stream` | POST | SSE streaming chat |
| `/jone/agent/jone/test` | POST | Direct operation test |
| `/jone/agent/jone/health` | GET | Agent health |
| `/jone/agent/jone/node/{node_id}` | GET | Get node by ID |
| `/jone/agent/jone/skill/{id}/tasks` | GET | Tasks for skill |

## Example Queries

```
"What tasks should someone with food safety knowledge do?"
"Who has project management skills in the infantry brigade?"
"Show readiness for all units with fill rate above 75%"
"Find personnel with expired weapons qualifications"
"Who is available for deployment from Pėstininkų brigada?"
"Write a job announcement for a warehouse worker"
```

## Development

### Adding New Operations

1. Add operation to `src/agent_jone/task_routing.json`
2. Implement function in `src/agent_jone/engine.py`
3. Add case to `execute_plan()` dispatcher
4. Update `JoneOp` enum in `config.py`
5. Update LLM prompt in `prompts.py` if needed

### Testing

```bash
# Run the test endpoint
curl -X POST http://localhost:8080/jone/agent/jone/test \
  -H "Content-Type: application/json" \
  -d '{"op": "fe_readiness", "params": {"limit": 10}}'
```

## License

Internal SAP use only.
