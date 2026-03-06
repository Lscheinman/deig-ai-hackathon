# src/agent_jone/plan_validation.py
"""
Plan validation and normalization for JONE agent.
Ensures LLM plans conform to allowed operations and limits.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from loguru import logger


def validate_and_normalize_plan(
    intent_obj: Dict[str, Any],
    *,
    rules: Dict[str, Any],
    task_routing: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Validate and normalize an LLM-generated plan.
    
    Args:
        intent_obj: Dict containing 'plan' key with {op, params}
        rules: Agent rules with defaults
        task_routing: Allowed operations and schemas
    
    Returns:
        Tuple of (normalized_plan, validation_debug)
    """
    defaults = rules.get("defaults") or {}
    max_limit = int(defaults.get("max_limit", 200))
    default_limit = int(defaults.get("default_limit", 50))
    
    plan = intent_obj.get("plan") or {}
    op = str(plan.get("op") or "").strip()
    params = dict(plan.get("params") or {})
    
    vdebug = {
        "raw_op": op,
        "raw_params": params.copy(),
        "adjustments": [],
    }
    
    # Validate op exists in task_routing
    allowed_ops = set((task_routing.get("ops") or {}).keys())
    if op not in allowed_ops:
        vdebug["adjustments"].append(f"Unknown op '{op}', defaulting to nl_search")
        op = "nl_search"
        if "query" not in params and "text_query" in params:
            params["query"] = params.pop("text_query")
    
    # Clamp limit parameters
    for limit_key in ("limit", "limit_tasks", "limit_candidates", "limit_qualities_per_task"):
        if limit_key in params:
            original = params[limit_key]
            clamped = _clamp_limit(original, default=default_limit, max_val=max_limit)
            if clamped != original:
                vdebug["adjustments"].append(f"{limit_key}: {original} → {clamped}")
            params[limit_key] = clamped
    
    # Ensure required defaults
    if "limit" not in params:
        params["limit"] = default_limit
    
    # Validate fuzzy params if present
    if "fuzzy" in params and isinstance(params["fuzzy"], dict):
        fuzzy = params["fuzzy"]
        if "max_candidates" in fuzzy:
            fuzzy["max_candidates"] = _clamp_limit(fuzzy["max_candidates"], default=100, max_val=max_limit)
        if "threshold" in fuzzy:
            try:
                fuzzy["threshold"] = max(0.0, min(float(fuzzy["threshold"]), 9999.0))
            except (ValueError, TypeError):
                fuzzy["threshold"] = 2.5
    
    normalized = {"op": op, "params": params}
    vdebug["normalized_op"] = op
    vdebug["normalized_params"] = params
    
    logger.debug(f"[plan_validation] Normalized: {normalized}")
    return normalized, vdebug


def _clamp_limit(val: Any, *, default: int, max_val: int) -> int:
    """Clamp a limit value to valid range."""
    try:
        v = int(val)
        return max(1, min(v, max_val))
    except (ValueError, TypeError):
        return default


def normalize_output_mode(intent_obj: Dict[str, Any]) -> str:
    """
    Normalize output_mode from LLM response.
    
    Args:
        intent_obj: Dict containing 'output_mode' key
    
    Returns:
        Normalized output mode: 'analytics', 'graph', or 'map'
    """
    raw = str(intent_obj.get("output_mode") or "").strip().lower()
    
    if raw in ("analytics", "table", "chart", "data"):
        return "analytics"
    if raw in ("graph", "network", "subgraph", "nodes"):
        return "graph"
    if raw in ("map", "geo", "location"):
        return "map"
    
    # Default based on plan operation
    plan = intent_obj.get("plan") or {}
    op = str(plan.get("op") or "").lower()
    
    if op in ("neighbors_subgraph", "shortest_path"):
        return "graph"
    
    return "analytics"
