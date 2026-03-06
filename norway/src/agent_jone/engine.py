# app/services/agent_jone/engine.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from loguru import logger
import pandas as pd

from hana_ml.graph import Graph
import hana_ml.graph.algorithms as hga

from commons.hana.hana_conn import get_hanaml_connection 


NodeType = Literal["qualification", "task", "quality"]
Direction = Literal["OUTGOING", "INCOMING", "ANY"]


# -----------------------------
# Config (keep it tiny for now)
# -----------------------------
DEFAULT_CFG: Dict[str, Any] = {
    "schema": "DFS_HR",
    "nodes_table": "SKILLS_AND_QUALS_NODES_NOR",
    "edges_table": "SKILLS_AND_QUALS_EDGES_NOR",
    "graph_workspace_schema": "DFS_HR",
    "graph_workspace_name": "GW_SKILLS",
    # columns
    "node_id_col": "NODE_ID",
    "node_type_col": "TYPE",
    "node_text_col": "TEXT_NO",
    "edge_source_col": "SOURCE",
    "edge_target_col": "TARGET",
    "edge_type_col": "EDGE_TYPE",
    "edge_desc_col": "DESCRIPTION",
}





def _cfg_merge(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(DEFAULT_CFG)
    if isinstance(cfg, dict):
        out.update(cfg)
    return out


# -----------------------------
# HANA bindings
# -----------------------------
def get_cc(*, autocommit: bool = True):
    """Factory hook; API can override and pass cc directly if it prefers."""
    return get_hanaml_connection(autocommit=autocommit)


def get_graph(cc, *, cfg: Optional[Dict[str, Any]] = None) -> Graph:
    c = _cfg_merge(cfg)
    return Graph(
        connection_context=cc,
        workspace_name=c["graph_workspace_name"],
        schema=c["graph_workspace_schema"],
    )


def get_nodes_df(cc, *, cfg: Optional[Dict[str, Any]] = None):
    c = _cfg_merge(cfg)
    return cc.table(c["nodes_table"], schema=c["schema"])


def get_edges_df(cc, *, cfg: Optional[Dict[str, Any]] = None):
    c = _cfg_merge(cfg)
    return cc.table(c["edges_table"], schema=c["schema"])


# -----------------------------
# Small safe helpers
# -----------------------------
def _escape_sql_literal(s: str) -> str:
    return (s or "").replace("'", "''")


def _in_list_expr(col: str, values: List[str]) -> str:
    """Build an IN (...) expression for hana_ml.DataFrame.filter()."""
    vals = [f"'{_escape_sql_literal(v)}'" for v in values if v is not None and str(v).strip()]
    if not vals:
        return "1=0"
    return f'"{col}" IN ({", ".join(vals)})'


def _detect_vertex_key_col(vertices_df: pd.DataFrame) -> str:
    """
    Neighbors/ShortestPath result frames vary in the vertex id column name across versions/workspaces.
    Try common candidates; fallback to first column.
    """
    for c in ("ID", "id", "VERTEX_KEY", "vertex_key", "NODE_ID", "node_id"):
        if c in vertices_df.columns:
            return c
    return vertices_df.columns[0]


# -----------------------------
# Simple table ops (cheap)
# -----------------------------
def search_nodes(
    cc,
    *,
    text_query: str,
    node_type: Optional[NodeType] = None,
    limit: int = 25,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Cheap node lookup by text. Uses hana_ml DataFrame filtering (compiled to SQL by hana_ml).
    """
    c = _cfg_merge(cfg)
    q = (text_query or "").strip().replace("%", "")
    if not q:
        return pd.DataFrame(columns=[c["node_id_col"], c["node_type_col"], c["node_text_col"]])

    df = get_nodes_df(cc, cfg=c)
    expr = f'LOWER("{c["node_text_col"]}") LIKE LOWER(\'%{_escape_sql_literal(q)}%\')'
    df = df.filter(expr)

    if node_type:
        df = df.filter(f'"{c["node_type_col"]}" = \'{_escape_sql_literal(node_type)}\'')

    return (
        df.select(c["node_id_col"], c["node_type_col"], c["node_text_col"])
        .head(int(limit))
        .collect()
    )


def get_node(
    cc,
    *,
    node_id: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    c = _cfg_merge(cfg)
    df = (
        get_nodes_df(cc, cfg=c)
        .filter(f'"{c["node_id_col"]}" = \'{_escape_sql_literal(node_id)}\'')
        .select(c["node_id_col"], c["node_type_col"], c["node_text_col"])
        .head(1)
        .collect()
    )
    if df.empty:
        return None
    row = df.iloc[0].to_dict()
    return {"node_id": row[c["node_id_col"]], "type": row[c["node_type_col"]], "text": row[c["node_text_col"]]}


def get_edges_for_node(
    cc,
    *,
    node_id: str,
    direction: Literal["out", "in", "any"] = "any",
    edge_type: Optional[str] = None,
    limit: int = 200,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    c = _cfg_merge(cfg)
    df = get_edges_df(cc, cfg=c)
    nid = _escape_sql_literal(node_id)

    if direction == "out":
        df = df.filter(f'"{c["edge_source_col"]}" = \'{nid}\'')
    elif direction == "in":
        df = df.filter(f'"{c["edge_target_col"]}" = \'{nid}\'')
    else:
        df = df.filter(
            f'"{c["edge_source_col"]}" = \'{nid}\' OR "{c["edge_target_col"]}" = \'{nid}\''
        )

    if edge_type:
        df = df.filter(f'"{c["edge_type_col"]}" = \'{_escape_sql_literal(edge_type)}\'')

    return (
        df.select(c["edge_source_col"], c["edge_target_col"], c["edge_type_col"], c["edge_desc_col"])
        .head(int(limit))
        .collect()
    )


# -----------------------------
# Graph algorithm ops (hana_ml)
# -----------------------------
def neighbors_vertices(
    cc,
    *,
    start_vertex: str,
    lower_bound: int = 1,
    upper_bound: int = 1,
    direction: Direction = "OUTGOING",
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Uses hana_ml.graph.algorithms.Neighbors.execute()
    """
    g = get_graph(cc, cfg=cfg)
    nb = hga.Neighbors(graph=g).execute(
        start_vertex=str(start_vertex),
        direction=direction,
        lower_bound=int(lower_bound),
        upper_bound=int(upper_bound),
    )
    return nb.vertices


def neighbors_subgraph(
    cc,
    *,
    start_vertex: str,
    lower_bound: int = 1,
    upper_bound: int = 1,
    direction: Direction = "OUTGOING",
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Uses hana_ml.graph.algorithms.NeighborsSubgraph.execute()
    Returns (vertices_df, edges_df).
    """
    g = get_graph(cc, cfg=cfg)
    nb = hga.NeighborsSubgraph(graph=g).execute(
        start_vertex=str(start_vertex),
        direction=direction,
        lower_bound=int(lower_bound),
        upper_bound=int(upper_bound),
    )
    return nb.vertices, nb.edges


def shortest_path(
    cc,
    *,
    source: str,
    target: str,
    direction: Direction = "OUTGOING",
    weight_col: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Uses hana_ml.graph.algorithms.ShortestPath.execute()
    """
    g = get_graph(cc, cfg=cfg)
    sp = hga.ShortestPath(graph=g).execute(
        source=str(source),
        target=str(target),
        direction=direction,
        weight=weight_col,
    )
    # Convert Decimal to float for JSON serialization
    weight = float(sp.weight) if sp.weight is not None else None
    return {"weight": weight, "vertices": sp.vertices, "edges": sp.edges}


# -----------------------------
# Domain intents (your HR graph)
# -----------------------------
def tasks_for_skill(
    cc,
    *,
    skill_node_id: str,
    limit: int = 50,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    'What tasks should a person with this skill do?'
    1-hop OUTGOING neighbor traversal, then enrich via nodes table and filter type='task'.
    """
    c = _cfg_merge(cfg)
    verts = neighbors_vertices(
        cc,
        start_vertex=skill_node_id,
        lower_bound=1,
        upper_bound=1,
        direction="OUTGOING",
        cfg=c,
    )
    if verts.empty:
        return pd.DataFrame(columns=[c["node_id_col"], c["node_type_col"], c["node_text_col"]])

    key_col = _detect_vertex_key_col(verts)
    ids = verts[key_col].dropna().astype(str).unique().tolist()
    if not ids:
        return pd.DataFrame(columns=[c["node_id_col"], c["node_type_col"], c["node_text_col"]])

    nodes = (
        get_nodes_df(cc, cfg=c)
        .filter(_in_list_expr(c["node_id_col"], ids))
        .filter(f'"{c["node_type_col"]}" = \'task\'')
        .select(c["node_id_col"], c["node_type_col"], c["node_text_col"])
        .collect()
    )
    return nodes.head(int(limit))

def qualifications_for_task(
    cc,
    *,
    task_node_id: str,
    limit: int = 50,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    'What qualifications are required for this task?'
    Deterministic 1-hop INCOMING traversal:
      qualification -> task
    Then enrich via nodes table and filter type='qualification'.
    """
    c = _cfg_merge(cfg)

    verts = neighbors_vertices(
        cc,
        start_vertex=task_node_id,
        lower_bound=1,
        upper_bound=1,
        direction="INCOMING",
        cfg=c,
    )
    if verts.empty:
        return pd.DataFrame(columns=[c["node_id_col"], c["node_type_col"], c["node_text_col"]])

    key_col = _detect_vertex_key_col(verts)
    ids = verts[key_col].dropna().astype(str).unique().tolist()
    if not ids:
        return pd.DataFrame(columns=[c["node_id_col"], c["node_type_col"], c["node_text_col"]])

    nodes = (
        get_nodes_df(cc, cfg=c)
        .filter(_in_list_expr(c["node_id_col"], ids))
        .filter(f"\"{c['node_type_col']}\" = 'qualification'")
        .select(c["node_id_col"], c["node_type_col"], c["node_text_col"])
        .collect()
    )
    return nodes.head(int(limit))

def qualities_for_task(
    cc,
    *,
    task_node_id: str,
    limit: int = 50,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    'What personal qualities best fit this task?'
    Cheapest path: edge-table filter on edge_type='quality_suits_task', then join to nodes table.
    """
    c = _cfg_merge(cfg)
    e = (
        get_edges_df(cc, cfg=c)
        .filter(f'"{c["edge_target_col"]}" = \'{_escape_sql_literal(task_node_id)}\'')
        .filter(f'"{c["edge_type_col"]}" = \'quality_suits_task\'')
        .select(c["edge_source_col"], c["edge_desc_col"])
        .head(500)
        .collect()
    )
    if e.empty:
        return pd.DataFrame(columns=[c["node_id_col"], c["node_type_col"], c["node_text_col"], "why"])

    q_ids = e[c["edge_source_col"]].astype(str).unique().tolist()
    q = (
        get_nodes_df(cc, cfg=c)
        .filter(_in_list_expr(c["node_id_col"], q_ids))
        .filter(f'"{c["node_type_col"]}" = \'quality\'')
        .select(c["node_id_col"], c["node_type_col"], c["node_text_col"])
        .collect()
    )

    out = q.merge(e, left_on=c["node_id_col"], right_on=c["edge_source_col"], how="left")
    out = out.rename(columns={c["edge_desc_col"]: "why"}).drop(columns=[c["edge_source_col"]])
    return out.head(int(limit))


def tasks_and_qualities_for_skill(
    cc,
    *,
    skill_node_id: str,
    limit_tasks: int = 50,
    limit_qualities_per_task: int = 5,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    2-hop answer object:
      qualification -> tasks
      then for each task -> qualities
    """
    c = _cfg_merge(cfg)
    skill = get_node(cc, node_id=skill_node_id, cfg=c)

    tasks = tasks_for_skill(cc, skill_node_id=skill_node_id, limit=limit_tasks, cfg=c)
    items: List[Dict[str, Any]] = []

    for _, row in tasks.iterrows():
        tid = str(row[c["node_id_col"]])
        qs = qualities_for_task(cc, task_node_id=tid, limit=limit_qualities_per_task, cfg=c)
        items.append(
            {
                "task": {"node_id": tid, "text": row[c["node_text_col"]]},
                "qualities": qs.to_dict(orient="records"),
            }
        )

    return {"skill": skill, "tasks": items}

# -----------------------------
# LLM plan execution (rails)
# -----------------------------
def execute_plan(
    cc,
    plan: Dict[str, Any],
    *,
    cfg: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Execute a small structured plan. Keep the LLM from inventing free-form queries.

    Example plans:
      {"op":"search_nodes","text_query":"matsikkerhet","node_type":"qualification","limit":10}
      {"op":"tasks_for_skill","skill_node_id":"N0003"}
      {"op":"qualities_for_task","task_node_id":"N0050"}
      {"op":"tasks_and_qualities_for_skill","skill_node_id":"N0003","limit_tasks":20,"limit_qualities_per_task":5}
      {"op":"neighbors_subgraph","start_vertex":"N0003","lower_bound":1,"upper_bound":2,"direction":"OUTGOING"}
      {"op":"shortest_path","source":"N0003","target":"N0100","direction":"ANY"}
    """
    c = _cfg_merge(cfg)
    op = (plan or {}).get("op")
    p = (plan or {}).get("params") or {}
    if not isinstance(p, dict):
        p = {}
    logger.info(f"[agent_jone][engine] execute_plan op={op} ")
    logger.info(f"[agent_jone][engine] execute_plan params_keys={list(p.keys())}")
    if not isinstance(plan, dict):
        raise ValueError("plan must be an object")

    op = plan.get("op")
    logger.info(f"[agent_jone][engine] execute_plan op={op}, keys={list(plan.keys())}")

    if op == "search_nodes":
        return search_nodes(
            cc,
            text_query=plan.get("text_query", ""),
            node_type=plan.get("node_type"),
            limit=int(plan.get("limit", 25)),
            cfg=c,
        )
    if op == "get_node":
        return get_node(cc, node_id=str(plan["node_id"]), cfg=c)
    if op == "get_edges_for_node":
        return get_edges_for_node(
            cc,
            node_id=str(plan["node_id"]),
            direction=plan.get("direction", "any"),
            edge_type=plan.get("edge_type"),
            limit=int(plan.get("limit", 200)),
            cfg=c,
        )
    if op == "tasks_for_skill":
        return tasks_for_skill(cc, skill_node_id=str(plan["skill_node_id"]), limit=int(plan.get("limit", 50)), cfg=c)
    if op == "qualities_for_task":
        return qualities_for_task(cc, task_node_id=str(plan["task_node_id"]), limit=int(plan.get("limit", 50)), cfg=c)
    if op == "tasks_and_qualities_for_skill":
        return tasks_and_qualities_for_skill(
            cc,
            skill_node_id=str(plan["skill_node_id"]),
            limit_tasks=int(plan.get("limit_tasks", 50)),
            limit_qualities_per_task=int(plan.get("limit_qualities_per_task", 5)),
            cfg=c,
        )
    if op == "neighbors_vertices":
        return neighbors_vertices(
            cc,
            start_vertex=str(plan["start_vertex"]),
            lower_bound=int(plan.get("lower_bound", 1)),
            upper_bound=int(plan.get("upper_bound", 1)),
            direction=plan.get("direction", "OUTGOING"),
            cfg=c,
        )
    if op == "neighbors_subgraph":
        v, e = neighbors_subgraph(
            cc,
            start_vertex=str(plan["start_vertex"]),
            lower_bound=int(plan.get("lower_bound", 1)),
            upper_bound=int(plan.get("upper_bound", 1)),
            direction=plan.get("direction", "OUTGOING"),
            cfg=c,
        )
        return {"vertices": v, "edges": e}
    if op == "shortest_path":
        return shortest_path(
            cc,
            source=str(plan["source"]),
            target=str(plan["target"]),
            direction=plan.get("direction", "OUTGOING"),
            weight_col=plan.get("weight_col"),
            cfg=c,
        )
    if op == "nl_search":
        query = str(plan.get("query") or plan["params"].get("query") or "")
        kw = plan.get("keywords") or (plan.get("params") or {}).get("keywords") or []
        node_type_hint = (plan.get("params") or {}).get("node_type_hint")
        limit = int((plan.get("params") or {}).get("limit", 25))
        fuzzy = (plan.get("params") or {}).get("fuzzy") or {}
        max_candidates = int(fuzzy.get("max_candidates") or c.get("max_limit") or 200)

        logger.info(
            "[agent_jone][nl_search] query=%r kw_n=%d node_type_hint=%r limit=%s max_candidates=%s",
            query, len(kw), node_type_hint, limit, max_candidates
        )
        return nl_search(
            cc,
            query=query,
            keywords=kw,
            node_type_hint=node_type_hint,
            fuzzy=fuzzy,
            limit=limit,
            cfg=c,
        )
    if op == "resolve_and_expand":
        p = plan.get("params") or {}
        return resolve_and_expand(
            cc,
            query=str(p.get("query") or ""),
            keywords=p.get("keywords"),
            node_type_hint=p.get("node_type_hint"),
            fuzzy=p.get("fuzzy"),
            limit_candidates=int(p.get("limit_candidates", 10)),
            limit_tasks=int(p.get("limit_tasks", 50)),
            auto_pick_min_score=float(p.get("auto_pick_min_score", 5.0)),
            cfg=c,
        )
    if op == "qualifications_for_task":
        p = plan.get("params") or {}
        return qualifications_for_task(
            cc,
            task_node_id=str(p.get("task_node_id") or ""),
            limit=int(p.get("limit", 50)),
            cfg=c,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Force Element Personnel Operations
    # ─────────────────────────────────────────────────────────────────────────
    if op == "personnel_for_fe":
        p = plan.get("params") or {}
        return personnel_for_fe(
            cc,
            fe_id=str(p.get("fe_id") or ""),
            limit=int(p.get("limit", 50)),
        )
    if op == "fe_for_personnel":
        p = plan.get("params") or {}
        return fe_for_personnel(
            cc,
            person_id=str(p.get("person_id") or ""),
        )
    if op == "position_status":
        p = plan.get("params") or {}
        return position_status(
            cc,
            fe_id=str(p.get("fe_id") or ""),
            include_details=bool(p.get("include_details", True)),
        )
    if op == "fe_readiness":
        p = plan.get("params") or {}
        return fe_readiness(
            cc,
            fe_id=p.get("fe_id"),  # Can be None for all FEs
            min_fill_rate=p.get("min_fill_rate"),
            limit=int(p.get("limit", 50)),
        )
    if op == "search_fe_personnel":
        p = plan.get("params") or {}
        return search_fe_personnel(
            cc,
            name_query=p.get("name") or p.get("name_query"),
            role_type=p.get("role_type"),
            military_grade=p.get("grade") or p.get("military_grade"),
            status=p.get("status"),
            limit=int(p.get("limit", 50)),
        )
    if op == "fe_personnel_by_competency":
        p = plan.get("params") or {}
        return fe_personnel_by_competency(
            cc,
            competency_query=p.get("competency_query") or p.get("competency") or p.get("skill"),
            fe_id=p.get("fe_id"),
            fe_name=p.get("fe_name") or p.get("unit_name") or p.get("unit"),
            min_rating=float(p.get("min_rating", 0.0)),
            limit=int(p.get("limit", 50)),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Military HR Operations
    # ─────────────────────────────────────────────────────────────────────────
    if op == "personnel_training":
        p = plan.get("params") or {}
        return personnel_training(
            cc,
            user_id=p.get("user_id") or p.get("person_id"),
            fe_name=p.get("fe_name") or p.get("unit_name") or p.get("unit"),
            training_type=p.get("training_type"),
            status=p.get("status"),
            limit=int(p.get("limit", 50)),
        )
    if op == "personnel_qualifications":
        p = plan.get("params") or {}
        return personnel_qualifications(
            cc,
            user_id=p.get("user_id") or p.get("person_id"),
            fe_name=p.get("fe_name") or p.get("unit_name") or p.get("unit"),
            qual_type=p.get("qual_type") or p.get("qualification_type"),
            status=p.get("status"),
            limit=int(p.get("limit", 50)),
        )
    if op == "personnel_deployments":
        p = plan.get("params") or {}
        return personnel_deployments(
            cc,
            user_id=p.get("user_id") or p.get("person_id"),
            fe_name=p.get("fe_name") or p.get("unit_name") or p.get("unit"),
            deployment_type=p.get("deployment_type"),
            status=p.get("status"),
            limit=int(p.get("limit", 50)),
        )
    if op == "personnel_medical":
        p = plan.get("params") or {}
        return personnel_medical(
            cc,
            user_id=p.get("user_id") or p.get("person_id"),
            fe_name=p.get("fe_name") or p.get("unit_name") or p.get("unit"),
            record_type=p.get("record_type"),
            deployable_only=bool(p.get("deployable_only", False)),
            limit=int(p.get("limit", 50)),
        )
    if op == "training_gaps":
        p = plan.get("params") or {}
        return training_gaps(
            cc,
            fe_name=p.get("fe_name") or p.get("unit_name") or p.get("unit"),
            training_type=p.get("training_type"),
            limit=int(p.get("limit", 50)),
        )
    if op == "qualification_status":
        p = plan.get("params") or {}
        return qualification_status(
            cc,
            fe_name=p.get("fe_name") or p.get("unit_name") or p.get("unit"),
            qual_type=p.get("qual_type") or p.get("qualification_type"),
            expired_only=bool(p.get("expired_only", False)),
            limit=int(p.get("limit", 50)),
        )
    if op == "deployment_availability":
        p = plan.get("params") or {}
        return deployment_availability(
            cc,
            fe_name=p.get("fe_name") or p.get("unit_name") or p.get("unit"),
            limit=int(p.get("limit", 50)),
        )

    raise ValueError(f"Unknown op: {op}")


def nl_search(
    cc,
    *,
    query: str,
    keywords: Optional[List[str]] = None,
    node_type_hint: Optional[NodeType] = None, # type: ignore
    fuzzy: Optional[Dict[str, Any]] = None,
    limit: int = 25,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Natural language search over nodes table with deterministic scoring.

    v1 approach (safe + simple):
    - Use LIKE-based matching for query + keywords (OR).
    - Pull a bounded candidate set (max_candidates).
    - Score in Python to avoid complex SQL.
    - Return top N.

    Later upgrades:
    - replace LIKE with CONTAINS / fulltext index
    - replace scoring with HANA fuzzy search functions
    """
    c = _cfg_merge(cfg)
    q = (query or "").strip()
    kw = [k.strip() for k in (keywords or []) if isinstance(k, str) and k.strip()]

    # Ensure primary query is included
    if q and q not in kw:
        kw = [q] + kw
    # De-dupe, keep order
    seen = set()
    kw = [k for k in kw if not (k.lower() in seen or seen.add(k.lower()))]

    if not kw:
        return pd.DataFrame(columns=[c["node_id_col"], c["node_type_col"], c["node_text_col"], "score", "matched_keywords"])

    # fuzzy knobs (bounded)
    fuzzy = fuzzy or {}
    max_candidates = int(fuzzy.get("max_candidates") or c.get("max_limit") or 200)
    max_candidates = max(1, min(max_candidates, int(c.get("max_limit", 200))))

    limit = max(1, min(int(limit), int(c.get("max_limit", 200))))

    nodes = get_nodes_df(cc, cfg=c)

    # Build a safe OR expression using LIKE
    # (Escaping % from user keywords, and quotes)
    like_clauses = []
    for k in kw:
        k2 = k.replace("%", "").replace("_", "")
        k2 = _escape_sql_literal(k2)
        like_clauses.append(f'LOWER("{c["node_text_col"]}") LIKE LOWER(\'%{k2}%\')')

    expr = " OR ".join(like_clauses)
    df = nodes.filter(expr)

    if node_type_hint:
        df = df.filter(f'"{c["node_type_col"]}" = \'{_escape_sql_literal(node_type_hint)}\'')

    # Pull candidates (bounded) and score locally
    cand = (
        df.select(c["node_id_col"], c["node_type_col"], c["node_text_col"])
        .head(max_candidates)
        .collect()
    )
    if cand.empty:
        return pd.DataFrame(columns=[c["node_id_col"], c["node_type_col"], c["node_text_col"], "score", "matched_keywords"])

    def _score(text: str) -> tuple[float, List[str]]:
        t = (text or "").lower()
        matched = []
        score = 0.0
        for k in kw:
            kl = k.lower()
            if not kl:
                continue
            if kl in t:
                matched.append(k)
                # heavier weight for longer phrases
                score += 2.0 + min(len(kl) / 10.0, 3.0)
                # bonus if it matches as a "whole word-ish"
                if f" {kl} " in f" {t} ":
                    score += 1.0
        # small bonus for matching multiple distinct keywords
        score += max(0, len(matched) - 1) * 0.75
        return score, matched

    scores = []
    matched_list = []
    for txt in cand[c["node_text_col"]].astype(str).tolist():
        s, m = _score(txt)
        scores.append(s)
        matched_list.append(", ".join(m))

    cand["score"] = scores
    cand["matched_keywords"] = matched_list

    # Apply optional fuzzy threshold (v1 uses score threshold)
    threshold = fuzzy.get("threshold")
    if threshold is not None:
        try:
            thr = float(threshold)
            cand = cand[cand["score"] >= thr]
        except Exception:
            pass

    cand = cand.sort_values(["score", c["node_type_col"], c["node_id_col"]], ascending=[False, True, True])
    return cand.head(limit).reset_index(drop=True)


def resolve_and_expand(
    cc,
    *,
    query: str,
    keywords: list[str] | None = None,
    node_type_hint: str | None = None,
    fuzzy: dict | None = None,
    limit_candidates: int = 10,
    limit_tasks: int = 50,
    auto_pick_min_score: float = 5.0,
    cfg: dict | None = None,
):
    """
    Deterministic 2-step:
      1) nl_search for candidates
      2) if best candidate strong enough (or only one), expand to tasks_for_skill / qualities_for_task
    """
    c = _cfg_merge(cfg)
    logger.info(f"[agent_jone][resolve_and_expand] query={query} kw={keywords} node_type_hint=%r limit_candidates=%s limit_tasks={limit_tasks} auto_pick_min_score=%s")
    cand_df = nl_search(
        cc,
        query=query,
        keywords=keywords,
        node_type_hint=node_type_hint,
        fuzzy=fuzzy,
        limit=limit_candidates,
        cfg=c,
    )

    out = {"candidates": cand_df, "expanded": None, "expanded_qualifications": None, "picked_node_id": None}


    if cand_df is None or cand_df.empty:
        return out

    # pick best
    logger.info(f"[agent_jone][resolve_and_expand] candidates n={len(cand_df)}")
    best = cand_df.iloc[0].to_dict()
    best_id = best.get(c["node_id_col"]) or best.get("NODE_ID")
    best_score = float(best.get("score") or 0.0)
    out["picked_node_id"] = best_id

    # expand if unambiguous enough
    if len(cand_df) == 1 or best_score >= float(auto_pick_min_score):
        if node_type_hint == "qualification":
            out["expanded"] = tasks_for_skill(
                cc,
                skill_node_id=str(best_id),
                limit=int(limit_tasks),
                cfg=c,
            )
        elif node_type_hint == "task":
            # keep expanded as qualities (existing behavior)
            ql = qualities_for_task(
                cc,
                task_node_id=str(best_id),
                limit=int(limit_tasks),
                cfg=c,
            )
            out["expanded"] = ql

            # New: also return qualifications for the task
            out["expanded_qualifications"] = qualifications_for_task(
                cc,
                task_node_id=str(best_id),
                limit=int(limit_tasks),
                cfg=c,
            )

        # else: leave expanded None
    logger.info(f"[agent_jone][resolve_and_expand] picked_node_id={out['picked_node_id']} best_score={best_score} expanded={out['expanded'] is not None}")
    return out


# -----------------------------
# Force Element (FE) Personnel Operations
# -----------------------------

def personnel_for_fe(
    cc,
    *,
    fe_id: str,
    include_vacant: bool = False,
    limit: int = 100,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Get personnel assigned to a Force Element.
    Uses V_FE_PERSONNEL_ROSTER view.
    
    Args:
        fe_id: Force Element ID (e.g., '50000034')
        include_vacant: If True, also return unfilled positions
        limit: Maximum rows to return
    
    Returns:
        DataFrame with personnel assignments
    """
    c = _cfg_merge(cfg)
    schema = c.get("schema", "DFS_HR")
    
    sql = f'''
        SELECT 
            FORCE_ELEMENT_ID,
            USER_ID,
            FIRST_NAME,
            LAST_NAME,
            NATIONALITY,
            ROLE_TYPE,
            POSITION_TITLE,
            MILITARY_GRADE,
            DUTY_TYPE,
            DUTY_STATUS,
            REPORT_DATE,
            DEROS,
            SF_JOB_TITLE,
            DEPARTMENT
        FROM "{schema}"."V_FE_PERSONNEL_ROSTER"
        WHERE FORCE_ELEMENT_ID = '{_escape_sql_literal(fe_id)}'
        ORDER BY MILITARY_GRADE, LAST_NAME
        LIMIT {int(limit)}
    '''
    
    return cc.sql(sql).collect()


def fe_for_personnel(
    cc,
    *,
    person_id: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get Force Element assignment for a person.
    
    Args:
        person_id: Person ID (e.g., 'P_DE_00001')
    
    Returns:
        Dict with FE assignment details or None if not assigned
    """
    c = _cfg_merge(cfg)
    schema = c.get("schema", "DFS_HR")
    
    sql = f'''
        SELECT 
            FORCE_ELEMENT_ID,
            USER_ID,
            FIRST_NAME,
            LAST_NAME,
            ROLE_TYPE,
            POSITION_TITLE,
            MILITARY_GRADE,
            DUTY_TYPE,
            DUTY_STATUS,
            REPORT_DATE,
            DEROS
        FROM "{schema}"."V_FE_PERSONNEL_ROSTER"
        WHERE USER_ID = '{_escape_sql_literal(person_id)}'
        LIMIT 1
    '''
    
    df = cc.sql(sql).collect()
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def position_status(
    cc,
    *,
    fe_id: str,
    include_details: bool = True,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get position fill status for a Force Element.
    Uses V_FE_POSITION_FILL and V_KEY_BILLET_STATUS views.
    
    Args:
        fe_id: Force Element ID
        include_details: If True, include position-level details
    
    Returns:
        Dict with summary and optional position details
    """
    c = _cfg_merge(cfg)
    schema = c.get("schema", "DFS_HR")
    
    # Get summary from V_FE_POSITION_FILL
    summary_sql = f'''
        SELECT 
            FORCE_ELEMENT_ID,
            AUTHORIZED_POSITIONS,
            FILLED_POSITIONS,
            VACANT_POSITIONS,
            FILL_RATE_PCT
        FROM "{schema}"."V_FE_POSITION_FILL"
        WHERE FORCE_ELEMENT_ID = '{_escape_sql_literal(fe_id)}'
    '''
    
    summary_df = cc.sql(summary_sql).collect()
    if summary_df.empty:
        return {"fe_id": fe_id, "found": False}
    
    summary = summary_df.iloc[0].to_dict()
    result = {
        "fe_id": fe_id,
        "found": True,
        "authorized_positions": int(summary.get("AUTHORIZED_POSITIONS", 0)),
        "filled_positions": int(summary.get("FILLED_POSITIONS", 0)),
        "vacant_positions": int(summary.get("VACANT_POSITIONS", 0)),
        "fill_rate_pct": float(summary.get("FILL_RATE_PCT", 0.0)),
    }
    
    if include_details:
        # Get key billet status
        billet_sql = f'''
            SELECT 
                POSITION_CODE,
                ROLE_TYPE,
                POSITION_TITLE,
                MILITARY_GRADE,
                REQUIRES_CLEARANCE,
                STATUS,
                INCUMBENT_USER_ID,
                INCUMBENT_NAME,
                DUTY_STATUS,
                REPORT_DATE,
                DEROS
            FROM "{schema}"."V_KEY_BILLET_STATUS"
            WHERE FORCE_ELEMENT_ID = '{_escape_sql_literal(fe_id)}'
            ORDER BY MILITARY_GRADE, ROLE_TYPE
        '''
        billet_df = cc.sql(billet_sql).collect()
        result["key_billets"] = billet_df.to_dict(orient="records")
    
    return result


def fe_readiness(
    cc,
    *,
    fe_id: Optional[str] = None,
    min_fill_rate: Optional[float] = None,
    limit: int = 50,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Get Force Element readiness summary.
    Uses V_FE_READINESS_SUMMARY view.
    
    Args:
        fe_id: Optional specific FE ID to query (if None, returns all)
        min_fill_rate: Optional minimum fill rate filter
        limit: Maximum rows to return
    
    Returns:
        DataFrame with readiness metrics
    """
    c = _cfg_merge(cfg)
    schema = c.get("schema", "DFS_HR")
    
    where_clauses = []
    if fe_id:
        where_clauses.append(f"FORCE_ELEMENT_ID = '{_escape_sql_literal(fe_id)}'")
    if min_fill_rate is not None:
        where_clauses.append(f"POSITION_FILL_RATE >= {float(min_fill_rate)}")
    
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    
    sql = f'''
        SELECT 
            FORCE_ELEMENT_ID,
            ASSIGNED_PERSONNEL,
            POSITION_FILL_RATE,
            KEY_BILLET_RATE,
            AVG_COMPETENCY_RATING,
            COMPETENCY_SCORE,
            READINESS_SCORE,
            READINESS_CATEGORY
        FROM "{schema}"."V_FE_READINESS_SUMMARY"
        {where_sql}
        ORDER BY READINESS_SCORE DESC
        LIMIT {int(limit)}
    '''
    
    return cc.sql(sql).collect()


def search_fe_personnel(
    cc,
    *,
    name_query: Optional[str] = None,
    role_type: Optional[str] = None,
    military_grade: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Search personnel across all Force Elements.
    
    Args:
        name_query: Search by first/last name (partial match)
        role_type: Filter by role type (e.g., 'Commander', 'XO')
        military_grade: Filter by grade (e.g., 'O-6', 'E-9')
        status: Filter by duty status (e.g., 'Present', 'Deployed', 'Leave')
        limit: Maximum rows to return
    
    Returns:
        DataFrame with matching personnel
    """
    c = _cfg_merge(cfg)
    schema = c.get("schema", "DFS_HR")
    
    where_clauses = []
    if name_query:
        q = _escape_sql_literal(name_query)
        where_clauses.append(
            f"(LOWER(FIRST_NAME) LIKE LOWER('%{q}%') OR LOWER(LAST_NAME) LIKE LOWER('%{q}%'))"
        )
    if role_type:
        where_clauses.append(f"ROLE_TYPE = '{_escape_sql_literal(role_type)}'")
    if military_grade:
        where_clauses.append(f"MILITARY_GRADE = '{_escape_sql_literal(military_grade)}'")
    if status:
        where_clauses.append(f"DUTY_STATUS = '{_escape_sql_literal(status)}'")
    
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    
    sql = f'''
        SELECT 
            FORCE_ELEMENT_ID,
            USER_ID,
            FIRST_NAME,
            LAST_NAME,
            NATIONALITY,
            ROLE_TYPE,
            POSITION_TITLE,
            MILITARY_GRADE,
            DUTY_TYPE,
            DUTY_STATUS,
            REPORT_DATE,
            DEROS
        FROM "{schema}"."V_FE_PERSONNEL_ROSTER"
        {where_sql}
        ORDER BY LAST_NAME, FIRST_NAME
        LIMIT {int(limit)}
    '''
    
    return cc.sql(sql).collect()


def fe_personnel_by_competency(
    cc,
    *,
    competency_query: str,
    fe_id: Optional[str] = None,
    fe_name: Optional[str] = None,
    min_rating: float = 0.0,
    limit: int = 50,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Find FE personnel with a specific competency.
    
    Args:
        competency_query: Competency name or code to search for 
                         (e.g., 'Project Management', 'COMP_PM', 'Leadership')
        fe_id: Optional Force Element ID to filter (e.g., '50000053')
               If None, searches all FEs
        fe_name: Optional Force Element name to filter (e.g., 'Pėstininkų brigada', 'Infantry')
               Uses fuzzy LIKE matching. If both fe_id and fe_name provided, fe_id takes precedence.
        min_rating: Minimum competency rating (0.0-5.0, default 0.0)
        limit: Maximum rows to return
    
    Returns:
        DataFrame with matching personnel and their competency details,
        including Force Element name for user-friendly display
    """
    c = _cfg_merge(cfg)
    schema = c.get("schema", "DFS_HR")
    
    q = _escape_sql_literal(competency_query)
    
    where_clauses = [
        f"""(
            LOWER(comp.NAME) LIKE LOWER('%{q}%') 
            OR LOWER(comp.NAME_DE) LIKE LOWER('%{q}%')
            OR LOWER(comp.NAME_LT) LIKE LOWER('%{q}%')
            OR comp.EXTERNAL_CODE = '{q}'
        )"""
    ]
    
    if min_rating > 0:
        where_clauses.append(f"ec.RATING >= {float(min_rating)}")
    
    if fe_id:
        where_clauses.append(f"pa.FORCE_ELEMENT_ID = '{_escape_sql_literal(fe_id)}'")
    elif fe_name:
        # Fuzzy match on Force Element name
        fn = _escape_sql_literal(fe_name)
        where_clauses.append(f"LOWER(fe.NAME) LIKE LOWER('%{fn}%')")
    
    where_sql = "WHERE " + " AND ".join(where_clauses)
    
    sql = f'''
        SELECT DISTINCT
            roster.FORCE_ELEMENT_ID,
            fe.NAME AS FORCE_ELEMENT_NAME,
            roster.USER_ID,
            roster.FIRST_NAME,
            roster.LAST_NAME,
            roster.ROLE_TYPE,
            roster.POSITION_TITLE,
            roster.MILITARY_GRADE,
            roster.DUTY_STATUS,
            comp.NAME AS COMPETENCY_NAME,
            ec.RATING AS COMPETENCY_RATING,
            ec.PROFICIENCY_LEVEL
        FROM "{schema}"."EMPCOMPETENCY" ec
        JOIN "{schema}"."COMPETENCY" comp ON ec.COMPETENCY_CODE = comp.EXTERNAL_CODE
        JOIN "{schema}"."FE_PERSONNEL_ASSIGNMENT" pa ON ec.USER_ID = pa.USER_ID
        JOIN "{schema}"."V_FE_PERSONNEL_ROSTER" roster ON pa.USER_ID = roster.USER_ID
            AND pa.FORCE_ELEMENT_ID = roster.FORCE_ELEMENT_ID
        LEFT JOIN "DFS_FE"."FE_NODES" fe ON CAST(pa.FORCE_ELEMENT_ID AS NVARCHAR(50)) = CAST(fe.ID AS NVARCHAR(50))
        {where_sql}
        ORDER BY ec.RATING DESC, roster.LAST_NAME
        LIMIT {int(limit)}
    '''
    
    return cc.sql(sql).collect()


# -----------------------------
# Military HR Operations
# -----------------------------
def personnel_training(
    cc,
    *,
    user_id: Optional[str] = None,
    fe_name: Optional[str] = None,
    training_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Get training records for personnel.
    
    Args:
        user_id: Specific person ID (e.g., 'P_LT_00201')
        fe_name: Force element name to filter by (e.g., 'Pėstininkų brigada')
        training_type: Filter by type (Course, Certification, Qualification)
        status: Filter by status (Completed, In Progress, Expired, Scheduled)
        limit: Max rows to return
    """
    c = _cfg_merge(cfg)
    schema = c.get("schema", "DFS_HR")
    
    where_clauses = []
    
    if user_id:
        where_clauses.append(f"t.USER_ID = '{_escape_sql_literal(user_id)}'")
    
    if fe_name:
        fn = _escape_sql_literal(fe_name)
        where_clauses.append(f"LOWER(fe.NAME) LIKE LOWER('%{fn}%')")
    
    if training_type:
        where_clauses.append(f"t.TRAINING_TYPE = '{_escape_sql_literal(training_type)}'")
    
    if status:
        where_clauses.append(f"t.STATUS = '{_escape_sql_literal(status)}'")
    
    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    
    sql = f'''
        SELECT 
            pp.FIRST_NAME,
            pp.LAST_NAME,
            fe.NAME AS FORCE_ELEMENT,
            t.TRAINING_NAME,
            t.TRAINING_TYPE,
            t.START_DATE,
            t.END_DATE,
            t.EXPIRY_DATE,
            t.STATUS,
            t.GRADE
        FROM "{schema}"."EMPTRAINING" t
        JOIN "{schema}"."PERPERSONAL" pp ON t.USER_ID = pp.PERSON_ID_EXTERNAL
        LEFT JOIN "{schema}"."FE_PERSONNEL_ASSIGNMENT" pa ON t.USER_ID = pa.USER_ID
        LEFT JOIN "DFS_FE"."FE_NODES" fe ON CAST(pa.FORCE_ELEMENT_ID AS NVARCHAR(50)) = CAST(fe.ID AS NVARCHAR(50))
        {where_sql}
        ORDER BY t.START_DATE DESC
        LIMIT {int(limit)}
    '''
    return cc.sql(sql).collect()


def personnel_qualifications(
    cc,
    *,
    user_id: Optional[str] = None,
    fe_name: Optional[str] = None,
    qual_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Get qualification records for personnel (weapons, vehicles, systems)."""
    c = _cfg_merge(cfg)
    schema = c.get("schema", "DFS_HR")
    
    where_clauses = []
    
    if user_id:
        where_clauses.append(f"q.USER_ID = '{_escape_sql_literal(user_id)}'")
    
    if fe_name:
        fn = _escape_sql_literal(fe_name)
        where_clauses.append(f"LOWER(fe.NAME) LIKE LOWER('%{fn}%')")
    
    if qual_type:
        where_clauses.append(f"q.QUAL_TYPE = '{_escape_sql_literal(qual_type)}'")
    
    if status:
        where_clauses.append(f"q.STATUS = '{_escape_sql_literal(status)}'")
    
    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    
    sql = f'''
        SELECT 
            pp.FIRST_NAME,
            pp.LAST_NAME,
            fe.NAME AS FORCE_ELEMENT,
            q.QUAL_NAME,
            q.QUAL_TYPE,
            q.QUAL_LEVEL,
            q.ISSUE_DATE,
            q.EXPIRY_DATE,
            q.STATUS,
            q.BADGE_EARNED
        FROM "{schema}"."EMPQUALIFICATION" q
        JOIN "{schema}"."PERPERSONAL" pp ON q.USER_ID = pp.PERSON_ID_EXTERNAL
        LEFT JOIN "{schema}"."FE_PERSONNEL_ASSIGNMENT" pa ON q.USER_ID = pa.USER_ID
        LEFT JOIN "DFS_FE"."FE_NODES" fe ON CAST(pa.FORCE_ELEMENT_ID AS NVARCHAR(50)) = CAST(fe.ID AS NVARCHAR(50))
        {where_sql}
        ORDER BY q.QUAL_TYPE, q.QUAL_NAME
        LIMIT {int(limit)}
    '''
    return cc.sql(sql).collect()


def personnel_deployments(
    cc,
    *,
    user_id: Optional[str] = None,
    fe_name: Optional[str] = None,
    deployment_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Get deployment history for personnel."""
    c = _cfg_merge(cfg)
    schema = c.get("schema", "DFS_HR")
    
    where_clauses = []
    
    if user_id:
        where_clauses.append(f"d.USER_ID = '{_escape_sql_literal(user_id)}'")
    
    if fe_name:
        fn = _escape_sql_literal(fe_name)
        where_clauses.append(f"LOWER(fe.NAME) LIKE LOWER('%{fn}%')")
    
    if deployment_type:
        where_clauses.append(f"d.DEPLOYMENT_TYPE = '{_escape_sql_literal(deployment_type)}'")
    
    if status:
        where_clauses.append(f"d.STATUS = '{_escape_sql_literal(status)}'")
    
    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    
    sql = f'''
        SELECT 
            pp.FIRST_NAME,
            pp.LAST_NAME,
            fe.NAME AS HOME_UNIT,
            d.DEPLOYMENT_NAME,
            d.DEPLOYMENT_TYPE,
            d.COUNTRY,
            d.START_DATE,
            d.END_DATE,
            d.STATUS,
            d.HAZARD_PAY,
            d.COMBAT_ZONE
        FROM "{schema}"."EMPDEPLOYMENT" d
        JOIN "{schema}"."PERPERSONAL" pp ON d.USER_ID = pp.PERSON_ID_EXTERNAL
        LEFT JOIN "{schema}"."FE_PERSONNEL_ASSIGNMENT" pa ON d.USER_ID = pa.USER_ID
        LEFT JOIN "DFS_FE"."FE_NODES" fe ON CAST(pa.FORCE_ELEMENT_ID AS NVARCHAR(50)) = CAST(fe.ID AS NVARCHAR(50))
        {where_sql}
        ORDER BY d.START_DATE DESC
        LIMIT {int(limit)}
    '''
    return cc.sql(sql).collect()


def personnel_medical(
    cc,
    *,
    user_id: Optional[str] = None,
    fe_name: Optional[str] = None,
    record_type: Optional[str] = None,
    deployable_only: bool = False,
    limit: int = 50,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Get medical readiness records for personnel."""
    c = _cfg_merge(cfg)
    schema = c.get("schema", "DFS_HR")
    
    where_clauses = []
    
    if user_id:
        where_clauses.append(f"m.USER_ID = '{_escape_sql_literal(user_id)}'")
    
    if fe_name:
        fn = _escape_sql_literal(fe_name)
        where_clauses.append(f"LOWER(fe.NAME) LIKE LOWER('%{fn}%')")
    
    if record_type:
        where_clauses.append(f"m.RECORD_TYPE = '{_escape_sql_literal(record_type)}'")
    
    if deployable_only:
        where_clauses.append("m.DEPLOYABLE = TRUE")
    
    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    
    sql = f'''
        SELECT 
            pp.FIRST_NAME,
            pp.LAST_NAME,
            fe.NAME AS FORCE_ELEMENT,
            m.RECORD_NAME,
            m.RECORD_TYPE,
            m.STATUS,
            m.EXAM_DATE,
            m.EXPIRY_DATE,
            m.RESULT,
            m.DEPLOYABLE,
            m.RESTRICTIONS
        FROM "{schema}"."EMPMEDICAL" m
        JOIN "{schema}"."PERPERSONAL" pp ON m.USER_ID = pp.PERSON_ID_EXTERNAL
        LEFT JOIN "{schema}"."FE_PERSONNEL_ASSIGNMENT" pa ON m.USER_ID = pa.USER_ID
        LEFT JOIN "DFS_FE"."FE_NODES" fe ON CAST(pa.FORCE_ELEMENT_ID AS NVARCHAR(50)) = CAST(fe.ID AS NVARCHAR(50))
        {where_sql}
        ORDER BY m.RECORD_TYPE, pp.LAST_NAME
        LIMIT {int(limit)}
    '''
    return cc.sql(sql).collect()


def training_gaps(
    cc,
    *,
    fe_name: Optional[str] = None,
    training_type: Optional[str] = None,
    limit: int = 50,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Find personnel with expired or missing training certifications."""
    c = _cfg_merge(cfg)
    schema = c.get("schema", "DFS_HR")
    
    where_clauses = ["t.STATUS = 'Expired'"]
    
    if fe_name:
        fn = _escape_sql_literal(fe_name)
        where_clauses.append(f"LOWER(fe.NAME) LIKE LOWER('%{fn}%')")
    
    if training_type:
        where_clauses.append(f"t.TRAINING_TYPE = '{_escape_sql_literal(training_type)}'")
    
    where_sql = "WHERE " + " AND ".join(where_clauses)
    
    sql = f'''
        SELECT 
            pp.FIRST_NAME,
            pp.LAST_NAME,
            fe.NAME AS FORCE_ELEMENT,
            t.TRAINING_NAME,
            t.TRAINING_TYPE,
            t.EXPIRY_DATE,
            t.STATUS,
            DAYS_BETWEEN(t.EXPIRY_DATE, CURRENT_DATE) AS DAYS_OVERDUE
        FROM "{schema}"."EMPTRAINING" t
        JOIN "{schema}"."PERPERSONAL" pp ON t.USER_ID = pp.PERSON_ID_EXTERNAL
        LEFT JOIN "{schema}"."FE_PERSONNEL_ASSIGNMENT" pa ON t.USER_ID = pa.USER_ID
        LEFT JOIN "DFS_FE"."FE_NODES" fe ON CAST(pa.FORCE_ELEMENT_ID AS NVARCHAR(50)) = CAST(fe.ID AS NVARCHAR(50))
        {where_sql}
        ORDER BY t.EXPIRY_DATE
        LIMIT {int(limit)}
    '''
    return cc.sql(sql).collect()


def qualification_status(
    cc,
    *,
    fe_name: Optional[str] = None,
    qual_type: Optional[str] = None,
    expired_only: bool = False,
    limit: int = 50,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Get qualification status, optionally filtering for expired ones."""
    c = _cfg_merge(cfg)
    schema = c.get("schema", "DFS_HR")
    
    where_clauses = []
    
    if fe_name:
        fn = _escape_sql_literal(fe_name)
        where_clauses.append(f"LOWER(fe.NAME) LIKE LOWER('%{fn}%')")
    
    if qual_type:
        where_clauses.append(f"q.QUAL_TYPE = '{_escape_sql_literal(qual_type)}'")
    
    if expired_only:
        where_clauses.append("q.STATUS = 'Expired'")
    
    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    
    sql = f'''
        SELECT 
            pp.FIRST_NAME,
            pp.LAST_NAME,
            fe.NAME AS FORCE_ELEMENT,
            q.QUAL_NAME,
            q.QUAL_TYPE,
            q.QUAL_LEVEL,
            q.EXPIRY_DATE,
            q.REQUALIFICATION_DUE,
            q.STATUS,
            CASE 
                WHEN q.STATUS = 'Expired' THEN DAYS_BETWEEN(q.EXPIRY_DATE, CURRENT_DATE)
                ELSE NULL 
            END AS DAYS_OVERDUE
        FROM "{schema}"."EMPQUALIFICATION" q
        JOIN "{schema}"."PERPERSONAL" pp ON q.USER_ID = pp.PERSON_ID_EXTERNAL
        LEFT JOIN "{schema}"."FE_PERSONNEL_ASSIGNMENT" pa ON q.USER_ID = pa.USER_ID
        LEFT JOIN "DFS_FE"."FE_NODES" fe ON CAST(pa.FORCE_ELEMENT_ID AS NVARCHAR(50)) = CAST(fe.ID AS NVARCHAR(50))
        {where_sql}
        ORDER BY q.STATUS DESC, q.EXPIRY_DATE
        LIMIT {int(limit)}
    '''
    return cc.sql(sql).collect()


def deployment_availability(
    cc,
    *,
    fe_name: Optional[str] = None,
    limit: int = 50,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Find personnel available for deployment (not currently deployed, 
    medically cleared, no expired critical qualifications).
    """
    c = _cfg_merge(cfg)
    schema = c.get("schema", "DFS_HR")
    
    where_clauses = []
    
    if fe_name:
        fn = _escape_sql_literal(fe_name)
        where_clauses.append(f"LOWER(fe.NAME) LIKE LOWER('%{fn}%')")
    
    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    
    sql = f'''
        SELECT 
            pp.FIRST_NAME,
            pp.LAST_NAME,
            fe.NAME AS FORCE_ELEMENT,
            pa.DUTY_STATUS,
            CASE 
                WHEN EXISTS (
                    SELECT 1 FROM "{schema}"."EMPDEPLOYMENT" d 
                    WHERE d.USER_ID = pp.PERSON_ID_EXTERNAL AND d.STATUS = 'Active'
                ) THEN 'Currently Deployed'
                WHEN EXISTS (
                    SELECT 1 FROM "{schema}"."EMPMEDICAL" m 
                    WHERE m.USER_ID = pp.PERSON_ID_EXTERNAL AND m.DEPLOYABLE = FALSE
                ) THEN 'Medical Hold'
                WHEN EXISTS (
                    SELECT 1 FROM "{schema}"."EMPQUALIFICATION" q 
                    WHERE q.USER_ID = pp.PERSON_ID_EXTERNAL AND q.STATUS = 'Expired' AND q.QUAL_TYPE = 'Weapon'
                ) THEN 'Weapons Qual Expired'
                WHEN EXISTS (
                    SELECT 1 FROM "{schema}"."EMPTRAINING" t 
                    WHERE t.USER_ID = pp.PERSON_ID_EXTERNAL AND t.STATUS = 'Expired' AND t.TRAINING_CODE = 'TRN_FIRST_AID'
                ) THEN 'First Aid Cert Expired'
                ELSE 'Available'
            END AS DEPLOYMENT_STATUS,
            (SELECT COUNT(*) FROM "{schema}"."EMPDEPLOYMENT" d WHERE d.USER_ID = pp.PERSON_ID_EXTERNAL) AS PAST_DEPLOYMENTS
        FROM "{schema}"."PERPERSONAL" pp
        JOIN "{schema}"."FE_PERSONNEL_ASSIGNMENT" pa ON pp.PERSON_ID_EXTERNAL = pa.USER_ID
        LEFT JOIN "DFS_FE"."FE_NODES" fe ON CAST(pa.FORCE_ELEMENT_ID AS NVARCHAR(50)) = CAST(fe.ID AS NVARCHAR(50))
        {where_sql}
        ORDER BY pp.LAST_NAME
        LIMIT {int(limit)}
    '''
    return cc.sql(sql).collect()
