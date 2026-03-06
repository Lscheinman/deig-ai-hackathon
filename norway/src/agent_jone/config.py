# src/agent_jone/config.py
"""
Configuration models and loaders for JONE agent.
"""
from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel

# Base directory for loading config files
_SRC_DIR = Path(__file__).parent


class JoneOp(str, Enum):
    """Available deterministic operations."""
    search_nodes = "search_nodes"
    get_node = "get_node"
    get_edges_for_node = "get_edges_for_node"
    tasks_for_skill = "tasks_for_skill"
    qualities_for_task = "qualities_for_task"
    tasks_and_qualities_for_skill = "tasks_and_qualities_for_skill"
    neighbors_subgraph = "neighbors_subgraph"
    shortest_path = "shortest_path"
    nl_search = "nl_search"
    resolve_and_expand = "resolve_and_expand"
    personnel_for_fe = "personnel_for_fe"
    fe_for_personnel = "fe_for_personnel"
    position_status = "position_status"
    fe_readiness = "fe_readiness"
    search_fe_personnel = "search_fe_personnel"
    fe_personnel_by_competency = "fe_personnel_by_competency"
    personnel_training = "personnel_training"
    personnel_qualifications = "personnel_qualifications"
    personnel_deployments = "personnel_deployments"
    personnel_medical = "personnel_medical"
    training_gaps = "training_gaps"
    qualification_status = "qualification_status"
    deployment_availability = "deployment_availability"


class JoneIntent(str, Enum):
    """Classified intent categories."""
    deterministic_call = "deterministic_call"
    capabilities = "capabilities"
    small_talk = "small_talk"
    out_of_scope = "out_of_scope"
    job_announcement = "job_announcement"


INTENTS = frozenset(e.value for e in JoneIntent)


class JoneAgentConfig(BaseModel):
    """Runtime configuration for the JONE agent."""
    target_schema: str = os.getenv("JONE_TARGET_SCHEMA", "DFS_HR")
    nodes_table: str = os.getenv("JONE_NODES_TABLE", "HR_NODES")
    edges_table: str = os.getenv("JONE_EDGES_TABLE", "HR_EDGES")
    graph_workspace_schema: str = os.getenv("JONE_GRAPH_SCHEMA", "DFS_HR")
    graph_workspace_name: str = os.getenv("JONE_GRAPH_WORKSPACE", "HR_GRAPH")
    max_limit: int = int(os.getenv("JONE_MAX_LIMIT", "200"))
    default_limit: int = int(os.getenv("JONE_DEFAULT_LIMIT", "50"))
    temperature: float = 0.0
    max_tokens: int = 512


class JoneTableResult(BaseModel):
    """Table result format."""
    columns: List[str]
    row_count: int
    rows: List[Dict[str, Any]]


class JoneGraphResult(BaseModel):
    """Graph result format."""
    nodes: JoneTableResult
    edges: JoneTableResult


class JoneTestRequest(BaseModel):
    """Test endpoint request."""
    op: JoneOp
    params: Optional[Dict[str, Any]] = None


class JoneTestResponse(BaseModel):
    """Test endpoint response."""
    ok: bool = True
    op: JoneOp
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    table: Optional[JoneTableResult] = None
    graph: Optional[JoneGraphResult] = None
    debug: Optional[Dict[str, Any]] = None


class JoneChatRequest(BaseModel):
    """Chat endpoint request."""
    query: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class JoneChatResponse(BaseModel):
    """Chat endpoint response."""
    intent: JoneIntent
    message: str
    output_mode: str = "analytics"
    analytics: Optional[Dict[str, Any]] = None
    graph: Optional[Dict[str, Any]] = None
    map: Optional[Dict[str, Any]] = None
    debug: Optional[Dict[str, Any]] = None


def load_agent_rules() -> Dict[str, Any]:
    """Load agent rules from rules.yaml."""
    rules_path = _SRC_DIR / "rules.yaml"
    if rules_path.exists():
        with open(rules_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {
        "defaults": {
            "max_limit": 200,
            "default_limit": 50,
            "confidence_threshold": 0.7
        }
    }


def load_task_routing() -> Dict[str, Any]:
    """Load task routing configuration from task_routing.json."""
    routing_path = _SRC_DIR / "task_routing.json"
    if routing_path.exists():
        with open(routing_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"ops": {}}
