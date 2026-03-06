# app/services/agent_jone/api.py
from __future__ import annotations

import json
from typing import Any, Dict, Optional, AsyncGenerator, Tuple

from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRouter
from loguru import logger
import asyncio

from dataclasses import dataclass
from typing import Callable, Awaitable, Optional, Any

from commons.ai_core.llm_client import get_llm_client
from commons.ai_core.llm_get_user_language import _LANG_RE
from commons.hana.hana_conn import get_hanaml_connection


from agent_jone.prompts import AGENT_SYSTEM_PROMPT
from agent_jone.plan_validation import (
    validate_and_normalize_plan,
    normalize_output_mode,
)

from commons.hana.hana_conn import get_hanaml_connection
from agent_jone.config import (
    JoneAgentConfig,
    JoneOp,
    JoneTestRequest,
    JoneTestResponse,
    JoneTableResult,
    JoneGraphResult,
    JoneChatRequest,
    JoneChatResponse,
    JoneIntent,
    load_agent_rules,
    load_task_routing,
    INTENTS
)

# deterministic engine functions (you already have these)
from agent_jone.engine import (
    search_nodes,
    get_node,
    tasks_for_skill,
    qualities_for_task,
    qualifications_for_task,
    neighbors_subgraph,
    shortest_path,
    execute_plan,
    fe_personnel_by_competency,
    fe_readiness,
)
from agent_jone.utils.sportsone_urls import inject_drill_down_urls_to_analytics

router = APIRouter(prefix="/agent/jone", tags=["Agent JOne"])
AGENT_KEY = "jone"


@dataclass
class _RunResult:
    intent: JoneIntent
    intent_label: str
    output_mode: str
    confidence: float
    threshold: float
    norm_op: str
    norm_params: dict
    result: Any
    analytics: Optional[dict] = None
    graph: Optional[dict] = None
    map_: Optional[dict] = None
    final_message: str = ""
    det_summary: str = ""
    vdebug: dict = None
    intent_obj: dict = None

@dataclass
class CoreResult:
    language_tag: str  # BCP-47-ish, e.g. "zh-Hans", "en", "nb", "und"
    intent: JoneIntent
    output_mode: str   # allow "text" too
    op: str
    params: dict
    deterministic_result: Any
    analytics: Optional[dict]
    graph: Optional[dict]
    map_: Optional[dict]
    message: str
    qc_fingerprint: str
    debug: dict


async def _write_job_announcement_with_llm(
    *,
    llm,
    language_tag: str,
    user_query: str,
    job_bundle: dict,
    max_chars: int = 3000,
) -> str:
    """
    LLM copywriter for job ads. Must not add facts beyond job_bundle.
    May include placeholders for unknown details.
    """
    system = (
        "You are a senior HR specialist writing a job posting.\n"
        "Write a natural, human job announcement.\n"
        "\n"
        "Hard rules:\n"
        "- Use ONLY the facts provided in job_bundle.\n"
        "- Do NOT invent benefits, salary, location, shift, company name, or requirements not listed.\n"
        "- If important details are missing, use clear placeholders like [Location], [Application deadline], [How to apply].\n"
        "- Keep NODE_IDs out of the ad body (they are internal). Use labels only.\n"
        f"- Output language: {language_tag}\n"
        "\n"
        "Structure:\n"
        "1) Title\n"
        "2) Short intro (2–3 sentences)\n"
        "3) Key responsibilities (bullets)\n"
        "4) Qualifications (bullets)\n"
        "5) Personal qualities (bullets)\n"
        "6) We offer (generic, non-specific: 'a professional environment', etc.)\n"
        "7) How to apply (placeholders)\n"
        "8) Equal opportunity line (generic)\n"
    )

    user = json.dumps(
        {"user_query": user_query, "job_bundle": job_bundle},
        ensure_ascii=False,
    )

    txt = await llm.generate_text(system=system, user=user, temperature=0.3, max_tokens=900)
    txt = (txt or "").strip()
    if len(txt) > max_chars:
        txt = txt[:max_chars].rstrip() + "…"
    return txt

def to_rtf(text: str) -> str:
    t = (text or "").replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    t = t.replace("\n", "\\par\n")
    return "{\\rtf1\\ansi\n" + t + "\n}"


def _build_job_ad_bundle_from_result(norm_op: str, result: Any) -> dict:
    """
    Extract a stable, minimal bundle for job-ad writing from deterministic outputs.
    Works for resolve_and_expand dict results and task_profile bundle.
    """
    bundle: dict = {
        "picked_task": None,            # {label, node_id}
        "alternative_tasks": [],        # [{label,node_id}]
        "qualities": [],                # [{label,node_id, why?}]
        "qualifications": [],           # [{label,node_id}]
    }

    if norm_op == "resolve_and_expand" and isinstance(result, dict):
        picked_id = result.get("picked_node_id")
        cand = result.get("candidates")
        exp = result.get("expanded")
        expq = result.get("expanded_qualifications")

        # candidates -> tasks
        if hasattr(cand, "to_dict"):
            rows = cand.to_dict(orient="records")
            for r in rows[:5]:
                label = (r.get("TEXT_NO") or r.get("TITLE") or "").strip()
                nid = (r.get("NODE_ID") or "").strip()
                if label and nid:
                    bundle["alternative_tasks"].append({"label": label, "node_id": nid})

        if picked_id:
            # pick label from candidates if possible
            picked_label = ""
            try:
                if hasattr(cand, "to_dict") and "NODE_ID" in cand.columns:
                    hit = cand[cand["NODE_ID"].astype(str) == str(picked_id)]
                    if getattr(hit, "shape", [0])[0] > 0:
                        picked_label = str(hit.iloc[0].get("TEXT_NO") or hit.iloc[0].get("TITLE") or "").strip()
            except Exception:
                pass
            bundle["picked_task"] = {"label": picked_label or str(picked_id), "node_id": str(picked_id)}

        # qualities
        if hasattr(exp, "to_dict"):
            for r in exp.to_dict(orient="records")[:10]:
                label = (r.get("TEXT_NO") or r.get("TITLE") or "").strip()
                nid = (r.get("NODE_ID") or "").strip()
                why = (r.get("why") or "").strip()
                if label and nid:
                    obj = {"label": label, "node_id": nid}
                    if why:
                        obj["why"] = why
                    bundle["qualities"].append(obj)

        # qualifications
        if hasattr(expq, "to_dict"):
            for r in expq.to_dict(orient="records")[:12]:
                label = (r.get("TEXT_NO") or r.get("TITLE") or "").strip()
                nid = (r.get("NODE_ID") or "").strip()
                if label and nid:
                    bundle["qualifications"].append({"label": label, "node_id": nid})

    return bundle


def _parse_jone_llm_json(raw: str) -> dict:
    """
    Agent Jone expects strict JSON from the model.

    This parser is intentionally agent-specific and lives in the agent layer.
    The shared LLM client returns raw text only.
    """
    if not raw or not isinstance(raw, str):
        return {}

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
        return {}
    except Exception:
        # log a short excerpt for debugging
        logger.warning("[agent_jone] LLM returned non-JSON output (first 300): %r", (raw or "")[:300])
        return {}


def _summarize_result_deterministic(op: str, result: Any, *, params: dict | None = None) -> str:
    """
    Build a user-facing closing message from deterministic results.

    This is intentionally deterministic and safe: no hallucinations.
    """
    params = params or {}

    if op == "resolve_and_expand" and isinstance(result, dict):
        picked = result.get("picked_node_id")
        cand = result.get("candidates")
        exp = result.get("expanded")
        expq = result.get("expanded_qualifications")  # optional, may be None

        cand_n = int(getattr(cand, "shape", [0])[0]) if cand is not None else 0
        exp_n = int(getattr(exp, "shape", [0])[0]) if exp is not None else 0
        expq_n = int(getattr(expq, "shape", [0])[0]) if expq is not None else 0

        q = (params.get("query") or "").strip()
        q_part = f' for "{q}"' if q else ""

        def _top_labels(df, *, max_items: int = 3) -> list[str]:
            if df is None or not hasattr(df, "to_dict") or getattr(df, "shape", [0])[0] == 0:
                return []
            cols = list(df.columns)
            text_col = "TEXT_NO" if "TEXT_NO" in cols else ("TITLE" if "TITLE" in cols else None)
            id_col = "NODE_ID" if "NODE_ID" in cols else None

            out = []
            for _, row in df.head(max_items).iterrows():
                label = str(row.get(text_col) or "").strip() if text_col else ""
                rid = str(row.get(id_col) or "").strip() if id_col else ""
                if label and rid:
                    out.append(f'{label} ({rid})')
                elif label:
                    out.append(label)
                elif rid:
                    out.append(rid)
            return out

        # Candidate tasks (evaluations)
        candidate_list = _top_labels(cand, max_items=3)

        # Related items (qualities) and qualifications (optional)
        related_qualities = _top_labels(exp, max_items=3)
        related_quals = _top_labels(expq, max_items=3)

        # Picked -> show picked task label if we can find it in candidates
        picked_label = ""
        try:
            if picked and cand is not None and hasattr(cand, "to_dict") and "NODE_ID" in cand.columns:
                hit = cand[cand["NODE_ID"].astype(str) == str(picked)]
                if getattr(hit, "shape", [0])[0] > 0:
                    picked_label = str(hit.iloc[0].get("TEXT_NO") or hit.iloc[0].get("TITLE") or "").strip()
        except Exception:
            pass

        if picked:
            parts = []
            if picked_label:
                parts.append(f'Mapped{q_part} to: "{picked_label}" ({picked}).')
            else:
                parts.append(f"Mapped{q_part} to node {picked}.")

            if related_qualities:
                parts.append("Related qualities: " + "; ".join(related_qualities) + ".")
            else:
                parts.append(f"Related qualities found: {exp_n} (capped).")

            if expq is not None:
                if related_quals:
                    parts.append("Related qualifications: " + "; ".join(related_quals) + ".")
                else:
                    parts.append(f"Related qualifications found: {expq_n} (capped).")

            if candidate_list:
                parts.append("Other close task matches: " + "; ".join(candidate_list) + ".")
            else:
                parts.append(f"Evaluated tasks: {cand_n}.")

            return " ".join(parts)

        # No pick -> candidates determine next step
        if cand_n == 0:
            return f'No matches found{q_part}. Try a more specific task label.'

        # Ambiguous: list candidates by name so user can choose without digging into NODE_IDs
        if candidate_list:
            return (
                f'Your query{q_part} is ambiguous. Top matches: '
                + "; ".join(candidate_list)
                + ". Reply with the best match name (or NODE_ID) and I’ll expand it."
            )

        return f'Your query{q_part} is ambiguous. Found {cand_n} candidates. Reply with the best match and I’ll expand it.'

    # nl_search returns a DataFrame
    if op == "nl_search":
        n = int(getattr(result, "shape", [0])[0]) if result is not None else 0
        if n <= 0:
            return "No matches found."

        # Prefer a human label if present
        label = ""
        try:
            top = result.iloc[0].to_dict()
            label = str(top.get("TEXT_NO") or top.get("TITLE") or top.get("TEXT") or "").strip()
        except Exception:
            pass

        if n == 1:
            return f"Found one match{(': “' + label + '”') if label else ''}. Expanding to related qualifications and qualities."
        return f"Found {n} possible matches. Narrow it down and I’ll expand to qualifications and qualities."


    if op == "tasks_for_skill":
        n = int(getattr(result, "shape", [0])[0]) if result is not None else 0
        return f"Returned **{n}** tasks for that skill (capped)."

    if op == "qualities_for_task":
        n = int(getattr(result, "shape", [0])[0]) if result is not None else 0
        return f"Returned **{n}** personal qualities for that task (capped)."

    # FE personnel competency search
    if op == "fe_personnel_by_competency":
        n = int(getattr(result, "shape", [0])[0]) if result is not None else 0
        comp_query = params.get("competency_query", "the requested competency")
        fe_id = params.get("fe_id")
        if n == 0:
            return f"Found no personnel with {comp_query} competency."
        
        # Try to extract top names and FE units for a helpful summary
        top_people = []
        try:
            for i, row in result.head(3).iterrows():
                name = f"{row.get('FIRST_NAME', '')} {row.get('LAST_NAME', '')}".strip()
                fe_name = row.get("FORCE_ELEMENT_NAME") or row.get("FORCE_ELEMENT_ID") or ""
                if name and fe_name:
                    top_people.append(f"{name} ({fe_name})")
                elif name:
                    top_people.append(name)
        except Exception:
            pass
        
        people_hint = ""
        if top_people:
            people_hint = f" Top matches: {', '.join(top_people)}."
        
        if fe_id:
            return f"Found **{n}** personnel with {comp_query} competency in the specified unit.{people_hint}"
        return f"Found **{n}** personnel with {comp_query} competency across all units.{people_hint}"

    # FE readiness
    if op == "fe_readiness":
        n = int(getattr(result, "shape", [0])[0]) if result is not None else 0
        return f"Returned **{n}** Force Element readiness records. Results shown in the analytics table."

    # graph-ish
    if op in ("neighbors_subgraph", "shortest_path"):
        return "Computed the requested graph result."

    return "Done."


def _safe_language_tag(tag: str | None) -> str:
    t = (tag or "").strip()[:20]
    if not t:
        return "und"
    if _LANG_RE.match(t) or t == "und":
        return t
    return "und"

async def _polish_summary_with_llm(
    *,
    llm,
    user_query: str,
    deterministic_summary: str,
    op: str,
    language_tag: str = "en",
    max_chars: int = 400,
    ui_policy: str = "cite_raw_labels",  # "cite_raw_labels" | "avoid_raw_labels"
) -> str:
    """
    LLM narrator only: rewrite deterministic summary into a nicer final message.
    Must not add facts beyond the deterministic summary.
    Narrative must be in the user's language (language_tag).
    Data payloads (tables/graph/map) must NOT be translated or altered.
    """
    async def _call(temp: float, tokens: int) -> str:
        txt = await llm.generate_text(
            system=system,
            user=user,
            temperature=temp,
            max_tokens=tokens,
        )
        return (txt or "").strip()

    def _clamp(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return det
        if len(s) > max_chars:
            return s[:max_chars].rstrip() + "…"
        return s

    def _violates_basic_rules(s: str) -> bool:
        # Cheap guardrails: block obvious bad behaviors.
        low = (s or "").lower()
        # "pick a node_id" UX smell
        if "pick a node" in low or "pick a node_id" in low or "choose a node_id" in low:
            return True
        # hallucination-y phrases (not perfect, but catches common failure modes)
        if "i found" in low and "deterministic summary" not in low and "found" not in det.lower():
            # if model adds extra findings not present in det_summary
            return True
        return False

    language_tag = _safe_language_tag(language_tag)

    # If deterministic summary is empty-ish, don't ask the LLM to invent a story.
    det = (deterministic_summary or "").strip()
    if not det:
        return ""

    system = (
        "You are Agent JOne, an HR graph assistant.\n"
        "Rewrite the deterministic summary into a friendly final message.\n"
        "\n"
        "Hard rules:\n"
        "- DO NOT add new facts beyond the deterministic summary.\n"
        "- DO NOT invent counts, node IDs, labels, or relationships.\n"
        "- Write the narrative fully in the user's language.\n"
        f"- User language tag: {language_tag}\n"
        "- Do NOT assume the data/table language matches the user language.\n"
        "- For skills graph queries: Keep NODE_IDs secondary (in parentheses). Ask for best match label (NODE_ID optional).\n"
        "- For Force Element / personnel queries: Do NOT mention NODE_ID. Use Force Element names.\n"
        "- NEVER ask the user for a NODE_ID. If clarification is needed, ask for unit name or person name.\n"
        f"- Max {max_chars} characters.\n"
        "\n"
        "Bilingual-avoidance rules (IMPORTANT):\n"
        "- If the deterministic summary contains labels in a different language than the user's language:\n"
        "  * Do NOT copy long foreign-language labels verbatim into the narrative.\n"
        "  * Instead, either:\n"
        "    (A) Provide a short translation/gloss for up to 2 labels, and cite the original label+NODE_ID briefly, OR\n"
        "    (B) Avoid labels entirely and point the user to the analytics table for exact labels.\n"
        f"- Use policy: {ui_policy}\n"
        "\n"
        "Output: a single short paragraph (no bullet lists).\n"
    )

    user = json.dumps(
        {
            "query": (user_query or "").strip(),
            "op": op,
            "deterministic_summary": det,
        },
        ensure_ascii=False,
    )

    try:
        # First attempt: normal polish
        txt = await _call(temp=0.2, tokens=160)
        txt = _clamp(txt)

        # One repair pass if it violated obvious rules
        if _violates_basic_rules(txt):
            repair_system = (
                system
                + "\nRepair instruction: Rewrite again obeying ALL hard rules. "
                  "Do not add facts. Do not ask user to pick NODE_ID; ask for name/label first.\n"
            )
            repair_user = json.dumps(
                {
                    "bad_draft": txt,
                    "query": (user_query or "").strip(),
                    "op": op,
                    "deterministic_summary": det,
                },
                ensure_ascii=False,
            )
            txt2 = await llm.generate_text(
                system=repair_system,
                user=repair_user,
                temperature=0.0,
                max_tokens=180,
            )
            txt2 = _clamp((txt2 or "").strip())
            return txt2 or det

        return txt or det

    except Exception as exc:
        logger.warning("[agent_jone] polish failed, using deterministic summary: %r", exc)
        return det
    

def _df_to_table(df) -> JoneTableResult:
    # df is a pandas DataFrame
    cols = list(df.columns)
    rows = df.to_dict(orient="records")
    return JoneTableResult(columns=cols, row_count=len(rows), rows=rows)


from decimal import Decimal


def _clean_for_json(obj):
    """Recursively convert Decimal to float for JSON serialization."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(i) for i in obj]
    return obj


def _df_to_analytics(df, *, name: str = "result") -> Dict[str, Any]:
    cols = list(df.columns)
    rows = df.to_dict(orient="records")
    # Clean Decimals for JSON serialization
    rows = [_clean_for_json(row) for row in rows]

    # Match the format used by Agent Chain/Chief that the UI understands
    return {
        "type": "table",
        "name": name,
        "columns": cols,
        "row_count": len(rows),
        "rows": rows,
    }


def _graph_to_payload(g: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize all graph outputs to:
      {"nodes": {"type":"table","name":"nodes","columns":[...],"row_count":N,"rows":[...]},
       "edges": {"type":"table","name":"edges","columns":[...],"row_count":N,"rows":[...]},
       "meta": {... optional ...}}
    """
    empty_table = {"type": "table", "name": "empty", "columns": [], "row_count": 0, "rows": []}
    if not isinstance(g, dict):
        return {"nodes": empty_table, "edges": empty_table, "meta": {}}

    # neighbors_subgraph: {"vertices": df, "edges": df}
    if "vertices" in g and "edges" in g and hasattr(g["vertices"], "to_dict"):
        return {
            "nodes": _df_to_analytics(g["vertices"], name="nodes"),
            "edges": _df_to_analytics(g["edges"], name="edges"),
            "meta": {},
        }

    # shortest_path: {weight, vertices, edges} where vertices/edges may already be dict-ish
    nodes = g.get("vertices")
    edges = g.get("edges")
    meta = {"weight": _clean_for_json(g.get("weight"))} if "weight" in g else {}

    if hasattr(nodes, "to_dict"):
        nodes = _df_to_analytics(nodes, name="nodes")
    if hasattr(edges, "to_dict"):
        edges = _df_to_analytics(edges, name="edges")

    if isinstance(nodes, dict) and "type" in nodes and isinstance(edges, dict) and "type" in edges:
        return {"nodes": nodes, "edges": edges, "meta": _clean_for_json(meta)}

    return {"nodes": empty_table, "edges": empty_table, "meta": _clean_for_json(meta)}


async def _llm_classify_and_plan(query: str, *, rules: dict, task_routing: dict, cfg) -> dict:
    llm = get_llm_client()

    user_payload = json.dumps(
        {"message": (query or "").strip(), "rules": rules, "task_routing": task_routing},
        ensure_ascii=False,
    )

    raw = await llm.generate_text(
        system=AGENT_SYSTEM_PROMPT,
        user=user_payload,
        temperature=getattr(cfg, "temperature", 0.0),
        max_tokens=getattr(cfg, "max_tokens", 512),
    )

    obj = _parse_jone_llm_json(raw)
    # Ensure language_tag exists and is valid-ish
    language_tag = (obj.get("language_tag") or "").strip()

    # If missing, use your existing LLM language detector module as fallback
    if not language_tag:
        try:
            # you already import _LANG_RE from commons.ai_core.llm_get_user_language
            # I’m assuming that module also provides a detector; if not, create one there.
            from commons.ai_core.llm_get_user_language import get_user_language_tag  # <-- adjust to real function
            language_tag = await get_user_language_tag(query)
        except Exception:
            language_tag = "und"

    language_tag = language_tag[:20] or "und"
    if not (_LANG_RE.match(language_tag) or language_tag == "und"):
        language_tag = "und"

    obj["language_tag"] = language_tag

    # HARD normalization here (agent-level)
    raw_intent = obj.get("intent")
    intent = (raw_intent.strip().lower() if raw_intent else "out_of_scope")
    if intent not in INTENTS:
        intent = "out_of_scope"
    obj["intent"] = intent

    # Logging: now accurate, no more client confusion
    logger.info(f"[agent_jone] LLM raw output (first 500): {raw[:500]}")
    logger.info(f"[agent_jone] LLM parsed intent: {intent}, confidence: {obj.get('confidence')}, ")

    return obj


def _capabilities_message(rules: dict, task_routing: dict) -> str:
    ops = sorted((task_routing.get("ops") or {}).keys())
    return (
        "I can search HR skills/tasks/qualities and traverse the HR graph deterministically.\n"
        f"Available ops: {', '.join(ops)}.\n"
        "Try: 'Find tasks for matsikkerhet' or 'Show qualities for task N0050' or 'Show related nodes around N0003'."
    )

# ----------------------------
# Streaming (SSE) surface
# ----------------------------
def _ts() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _envelope(*, type_: str, session_id: Optional[str], intent: Optional[str], payload: dict) -> dict:
    """Build SSE event envelope with camelCase keys for frontend compatibility."""
    out = {
        "type": type_,
        "requestId": session_id or "",
        "agentKey": AGENT_KEY,
        "intent": intent,
        "ts": _ts()
    }
    out.update(payload or {})
    return out

    
def _select_plan(intent_obj: dict, rules: dict) -> tuple[dict, float]:
    confidence = float(intent_obj.get("confidence") or 0.0)
    thr = float((rules.get("defaults") or {}).get("confidence_threshold") or 0.7)
    chosen = intent_obj.get("plan") if confidence >= thr else intent_obj.get("fallback_plan") or intent_obj.get("plan")
    if not isinstance(chosen, dict):
        chosen = {}
    return chosen, thr


def _pack_result(
    *,
    norm_op: str,
    result: Any,
    auto_profile: Optional[dict],
) -> tuple[Optional[dict], Optional[dict]]:
    """
    Returns (analytics, graph). Uses the exact shapes your UI expects today.
    Injects Sports One drill-down URLs for personnel-related operations.
    """
    if norm_op in ("neighbors_subgraph", "shortest_path"):
        return None, _graph_to_payload(result)

    if norm_op == "resolve_and_expand" and isinstance(result, dict):
        cand = result.get("candidates")
        exp = result.get("expanded")
        expq = result.get("expanded_qualifications")
        analytics = {
            "type": "bundle",
            "name": "resolve_and_expand",
            "picked_node_id": result.get("picked_node_id"),
            "candidates": _df_to_analytics(cand, name="candidates") if hasattr(cand, "to_dict") else None,
            "expanded": _df_to_analytics(exp, name="expanded") if hasattr(exp, "to_dict") else None,
            "expanded_qualifications": _df_to_analytics(expq, name="expanded_qualifications") if hasattr(expq, "to_dict") else None,
        }
        return analytics, None

    if auto_profile is not None:
        analytics = {
            "type": "bundle",
            "name": "task_profile",
            "matched_task": auto_profile["task"],
            "qualities": _df_to_analytics(auto_profile["qualities"], name="qualities")
            if hasattr(auto_profile["qualities"], "to_dict") else None,
            "qualifications": _df_to_analytics(auto_profile["qualifications"], name="qualifications")
            if hasattr(auto_profile["qualifications"], "to_dict") else None,
        }
        return analytics, None

    # Standard table result - inject drill-down URLs for personnel operations
    analytics = _df_to_analytics(result, name=norm_op)
    
    # Inject Sports One drill-down URLs for HR/personnel operations
    personnel_ops = {
        "fe_personnel_by_competency",
        "personnel_training",
        "personnel_qualifications", 
        "personnel_deployments",
        "personnel_medical",
        "training_gaps",
        "qualification_status",
        "deployment_availability",
        "fe_readiness",
    }
    if norm_op in personnel_ops:
        analytics = inject_drill_down_urls_to_analytics(analytics, entity_type="player")
    
    return analytics, None


def _should_fallback_resolve_and_expand(norm_op: str, result: Any) -> bool:
    if norm_op != "resolve_and_expand" or not isinstance(result, dict):
        return False
    picked = result.get("picked_node_id")
    cand = result.get("candidates")
    cand_n = int(getattr(cand, "shape", [0])[0]) if cand is not None else 0
    return (not picked) and cand_n == 0


async def _run_jone(
    *,
    req: JoneChatRequest,
    cfg: JoneAgentConfig,
    rules: dict,
    task_routing: dict,
    emit: Optional[Callable[[str, dict], Awaitable[None]]] = None,
) -> _RunResult:
    """
    Single source of truth for routing + execution + packing.
    emit(event_type, payload) is optional (used by stream_chat).
    """

    async def _emit(t: str, payload: dict):
        if emit is not None:
            await emit(t, payload)

    q = (req.query or "").strip()
    if not q:
        return _RunResult(
            intent=JoneIntent.out_of_scope,
            intent_label=JoneIntent.out_of_scope.value,
            output_mode="text",
            confidence=0.0,
            threshold=float((rules.get("defaults") or {}).get("confidence_threshold") or 0.7),
            norm_op="",
            norm_params={},
            result=None,
            final_message="Empty query.",
            vdebug={},
            intent_obj={},
        )

    # Truncate query for display (max 60 chars)
    query_preview = q[:57] + "..." if len(q) > 60 else q
    await _emit("progress", {"message": f"Analyzing: '{query_preview}'"})
    intent_obj = await _llm_classify_and_plan(q, cfg=cfg, rules=rules, task_routing=task_routing)

    intent_label = str(intent_obj.get("intent") or JoneIntent.out_of_scope.value)
    intent = JoneIntent(intent_label) if intent_label in (e.value for e in JoneIntent) else JoneIntent.out_of_scope
    output_mode = normalize_output_mode(intent_obj)
    reply = str(intent_obj.get("reply") or "").strip() or "OK."
    chosen, thr = _select_plan(intent_obj, rules)
    confidence = float(intent_obj.get("confidence") or 0.0)

    # LLM-chosen language tag (BCP-47-ish). Narrative uses this; data payload is untouched.
    language_tag = str(intent_obj.get("language_tag") or "").strip()[:20] or "und"
    if not (_LANG_RE.match(language_tag) or language_tag == "und"):
        language_tag = "und"

    # Emit meta early so stream can route subsequent events deterministically.
    await _emit("meta", {
        "message": "Intent ready.",
        "intent": intent_label,
        "output_mode": output_mode,
        "confidence": confidence,
        "threshold": thr,
        "language_tag": language_tag,
    })

    await _emit("bundle", {
        "message": "Plan built.",
        "llm": {
            "intent": intent_label,
            "confidence": confidence,
            "output_mode": output_mode,
            "language_tag": language_tag,
            "plan": intent_obj.get("plan"),
            "fallback_plan": intent_obj.get("fallback_plan"),
        }
    })
    
    # Non-deterministic intents
    if intent == JoneIntent.capabilities:
        msg = _capabilities_message(rules, task_routing)
        return _RunResult(intent=intent, intent_label=intent_label, output_mode=output_mode,
                          confidence=confidence, threshold=thr, norm_op="", norm_params={},
                          result=None, final_message=msg, vdebug={}, intent_obj=intent_obj)

    if intent == JoneIntent.small_talk:
        msg = "I’m here for HR graph questions. Ask about skills, tasks, and personal qualities."
        return _RunResult(intent=intent, intent_label=intent_label, output_mode=output_mode,
                          confidence=confidence, threshold=thr, norm_op="", norm_params={},
                          result=None, final_message=msg, vdebug={}, intent_obj=intent_obj)

    if intent == JoneIntent.out_of_scope:
        return _RunResult(intent=intent, intent_label=intent_label, output_mode=output_mode,
                          confidence=confidence, threshold=thr, norm_op="", norm_params={},
                          result=None, final_message=reply, vdebug={}, intent_obj=intent_obj)

    # Deterministic execution
    cc = get_hanaml_connection(autocommit=True)
    try:
        await _emit("progress", {"message": f"Planning {chosen.get('op', 'operation')}..."})
        norm_plan, vdebug = validate_and_normalize_plan({"plan": chosen}, rules=rules, task_routing=task_routing)

        # Build descriptive execution message
        op_display = norm_plan['op'].replace('_', ' ').title()
        params = norm_plan.get('params') or {}
        param_hints = []
        if params.get('competency_query'):
            param_hints.append(f"competency='{params['competency_query']}'")
        if params.get('fe_name'):
            param_hints.append(f"unit='{params['fe_name']}'")
        if params.get('query'):
            param_hints.append(f"query='{params['query'][:30]}...'" if len(str(params.get('query', ''))) > 30 else f"query='{params['query']}'")
        if params.get('node_id'):
            param_hints.append(f"node='{params['node_id']}'")
        if params.get('min_rating'):
            param_hints.append(f"min_rating={params['min_rating']}")
        
        exec_msg = f"Searching: {op_display}"
        if param_hints:
            exec_msg += f" ({', '.join(param_hints[:3])})"
        await _emit("progress", {"message": exec_msg, "op": norm_plan["op"], "params": norm_plan["params"]})
        result = execute_plan(cc, {"op": norm_plan["op"], "params": norm_plan["params"]}, cfg={
            "schema": cfg.target_schema,
            "nodes_table": cfg.nodes_table,
            "edges_table": cfg.edges_table,
            "graph_workspace_schema": cfg.graph_workspace_schema,
            "graph_workspace_name": cfg.graph_workspace_name,
            "max_limit": cfg.max_limit,
            "default_limit": cfg.default_limit,
        })

        # Deterministic fallback for resolve_and_expand empty
        if _should_fallback_resolve_and_expand(norm_plan["op"], result):
            fb = intent_obj.get("fallback_plan")
            if isinstance(fb, dict) and fb.get("op"):
                fb_op_display = fb.get('op', '').replace('_', ' ').title()
                await _emit("progress", {"message": f"No results, trying alternate search: {fb_op_display}..."})
                fb_norm, fb_vdebug = validate_and_normalize_plan({"plan": fb}, rules=rules, task_routing=task_routing)
                result = execute_plan(cc, {"op": fb_norm["op"], "params": fb_norm["params"]}, cfg={
                    "schema": cfg.target_schema,
                    "nodes_table": cfg.nodes_table,
                    "edges_table": cfg.edges_table,
                    "graph_workspace_schema": cfg.graph_workspace_schema,
                    "graph_workspace_name": cfg.graph_workspace_name,
                    "max_limit": cfg.max_limit,
                    "default_limit": cfg.default_limit,
                })
                norm_plan = fb_norm
                vdebug.update({"fallback_executed": True, "fallback_op": fb_norm["op"], **(fb_vdebug or {})})

        # Auto-expand nl_search single task hit -> task_profile bundle
        auto_profile = None
        try:
            if norm_plan["op"] == "nl_search" and result is not None and getattr(result, "shape", [0])[0] == 1:
                row0 = result.iloc[0].to_dict()
                node_id = str(row0.get("NODE_ID") or "").strip()
                node_type = str(row0.get("TYPE") or "").strip().lower()
                if node_id and node_type == "task":
                    q_df = qualities_for_task(cc, task_node_id=node_id, limit=50, cfg=cfg.model_dump())
                    req_df = qualifications_for_task(cc, task_node_id=node_id, limit=50, cfg=cfg.model_dump())
                    auto_profile = {"task": row0, "qualities": q_df, "qualifications": req_df}
                    vdebug.update({"auto_expanded_task_profile": True, "auto_task_node_id": node_id})
        except Exception as _exc:
            logger.warning("[agent_jone] auto-expand task profile failed: %r", _exc)

        # Get row count for progress message
        row_count = 0
        if result is not None:
            if hasattr(result, 'shape'):
                row_count = result.shape[0]
            elif hasattr(result, '__len__'):
                row_count = len(result)
        
        await _emit("progress", {"message": f"Found {row_count} result{'s' if row_count != 1 else ''}, processing..."})
        analytics, graph = _pack_result(norm_op=norm_plan["op"], result=result, auto_profile=auto_profile)

        await _emit("progress", {"message": f"Generating summary for {row_count} record{'s' if row_count != 1 else ''}..."})
        det_summary = _summarize_result_deterministic(norm_plan["op"], result, params=norm_plan.get("params") or {})
        
        if intent == JoneIntent.job_announcement:
            job_bundle = _build_job_ad_bundle_from_result(norm_plan["op"], result)

            # If we didn't confidently pick a task, ask one clarifying question label-first
            if not job_bundle.get("picked_task") and job_bundle.get("alternative_tasks"):
                top = job_bundle["alternative_tasks"][:2]
                opts = " vs ".join([f'"{x["label"]}"' for x in top])
                msg = (
                    f"I found multiple close warehouse-related task matches. Which one best fits the role: {opts}?\n"
                    "Reply with the best match name and I’ll generate the full job posting."
                )
                return _RunResult(
                    intent=intent,
                    intent_label=intent_label,
                    output_mode="analytics",
                    confidence=confidence,
                    threshold=thr,
                    norm_op=norm_plan["op"],
                    norm_params=norm_plan["params"],
                    result=result,
                    analytics=analytics,   # keep deterministic bundle available
                    graph=graph,
                    map_=None,
                    final_message=msg,
                    det_summary=det_summary,
                    vdebug=vdebug,
                    intent_obj=intent_obj,
                )

            ad = await _write_job_announcement_with_llm(
                llm=get_llm_client(),
                language_tag=language_tag,
                user_query=req.query,
                job_bundle=job_bundle,
            )

            return _RunResult(
                intent=intent,
                intent_label=intent_label,
                output_mode="analytics",
                confidence=confidence,
                threshold=thr,
                norm_op=norm_plan["op"],
                norm_params=norm_plan["params"],
                result=result,
                analytics=analytics,
                graph=graph,
                map_=None,
                final_message=ad,
                det_summary=det_summary,
                vdebug={**(vdebug or {}), "job_bundle": job_bundle},
                intent_obj=intent_obj,
            )

        # Finalize with language info
        lang_display = language_tag if language_tag != "und" else "auto"
        await _emit("progress", {"message": f"Writing response (lang={lang_display})..."})
        
        final_msg = await _polish_summary_with_llm(
            llm=get_llm_client(),
            user_query=req.query,
            deterministic_summary=det_summary,
            op=norm_plan["op"],
            language_tag=language_tag,
        )

        return _RunResult(
            intent=intent,
            intent_label=intent_label,
            output_mode=output_mode,
            confidence=confidence,
            threshold=thr,
            norm_op=norm_plan["op"],
            norm_params=norm_plan["params"],
            result=result,
            analytics=analytics,
            graph=graph,
            map_=None,
            final_message=final_msg,
            det_summary=det_summary,
            vdebug=vdebug,
            intent_obj=intent_obj,
        )

    finally:
        try:
            cc.close()
        except Exception:
            pass

async def stream_chat(req: JoneChatRequest) -> AsyncGenerator[dict, None]:
    cfg = JoneAgentConfig()
    rules = load_agent_rules()
    task_routing = load_task_routing()

    sid = req.session_id
    current_intent: Optional[str] = None

    # immediate heartbeat
    yield _envelope(type_="meta", session_id=sid, intent=None, payload={"message": "started"})

    q: "asyncio.Queue[Tuple[str, dict]]" = asyncio.Queue()

    async def emit_cb(t: str, p: dict):
        await q.put((t, p))

    run_task = asyncio.create_task(
        _run_jone(req=req, cfg=cfg, rules=rules, task_routing=task_routing, emit=emit_cb)
    )

    try:
        while True:
            if run_task.done() and q.empty():
                break
            try:
                t, payload = await asyncio.wait_for(q.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue

            if t == "meta" and isinstance(payload, dict) and payload.get("intent"):
                current_intent = str(payload["intent"])

            yield _envelope(type_=t, session_id=sid, intent=current_intent, payload=payload)

        run = await run_task

        if run.graph is not None:
            yield _envelope(type_="graph", session_id=sid, intent=run.intent_label, payload={"graph": run.graph})
        if run.analytics is not None:
            yield _envelope(type_="analytics", session_id=sid, intent=run.intent_label, payload={"spec": run.analytics})

        yield _envelope(
            type_="final",
            session_id=sid,
            intent=run.intent_label,
            payload={
                "message": run.final_message,
                "debug": {
                    "confidence": run.confidence,
                    "threshold": run.threshold,
                    "selected_op": run.norm_op,
                    "selected_params": run.norm_params,
                    "deterministic_summary": run.det_summary,
                    **(run.vdebug or {}),
                },
            },
        )

    except Exception as exc:
        logger.exception("[agent_jone] stream_chat failed: %r", exc)
        yield _envelope(type_="error", session_id=sid, intent=current_intent, payload={"message": "Execution failed", "error": repr(exc)})

    finally:
        yield _envelope(type_="done", session_id=sid, intent=current_intent, payload={})


async def chat(req: JoneChatRequest) -> JoneChatResponse:

    cfg = JoneAgentConfig()
    rules = load_agent_rules()
    task_routing = load_task_routing()

    run = await _run_jone(req=req, cfg=cfg, rules=rules, task_routing=task_routing)

    resp = JoneChatResponse(
        intent=run.intent,
        message=run.final_message,
        output_mode=run.output_mode,
        analytics=run.analytics,
        graph=run.graph,
        map=run.map_,
        debug={
            "intent_raw": run.intent_obj,
            "confidence": run.confidence,
            "threshold": run.threshold,
            "selected_op": run.norm_op,
            "selected_params": run.norm_params,
            "deterministic_summary": run.det_summary,
            **(run.vdebug or {}),
        },
    )
    return resp

# ----------------------------
# FastAPI endpoints
# ----------------------------
@router.post("/chat")
async def jone_chat(req: JoneChatRequest):
    resp = await chat(req)
    return JSONResponse(content=resp.model_dump())


@router.post("/stream")
async def jone_stream(req: JoneChatRequest):
    async def event_source():
        async for ev in stream_chat(req):
            yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
    return StreamingResponse(event_source(), media_type="text/event-stream")

@router.get("/health")
async def health():
    return JSONResponse(content={"ok": True, "agent": AGENT_KEY})


@router.post("/test")
async def test_call(req: JoneTestRequest):
    """
    One deterministic endpoint to rule them all.
    Useful for fast iteration in Swagger:
      {"op":"tasks_for_skill","params":{"skill_node_id":"N0003"}}
    """
    cfg = JoneAgentConfig()
    op = req.op
    params = req.params or {}

    logger.info("[agent_jone] test op={} params_keys={}", op.value, list(params.keys()))

    cc = get_hanaml_connection(autocommit=True)
    try:
        if op == JoneOp.search_nodes:
            df = search_nodes(
                cc,
                text_query=str(params.get("text_query", "")),
                node_type=params.get("node_type"),
                limit=int(params.get("limit", 25)),
                cfg=cfg.model_dump(),
            )
            return JoneTestResponse(op=op, message="OK", table=_df_to_table(df)).model_dump()

        if op == JoneOp.get_node:
            obj = get_node(
                cc,
                node_id=str(params["node_id"]),
                cfg=cfg.model_dump(),
            )
            return JoneTestResponse(op=op, message="OK", result={"node": obj}).model_dump()

        if op == JoneOp.tasks_for_skill:
            df = tasks_for_skill(
                cc,
                skill_node_id=str(params["skill_node_id"]),
                limit=int(params.get("limit", 50)),
                cfg=cfg.model_dump(),
            )
            return JoneTestResponse(op=op, message="OK", table=_df_to_table(df)).model_dump()

        if op == JoneOp.qualities_for_task:
            df = qualities_for_task(
                cc,
                task_node_id=str(params["task_node_id"]),
                limit=int(params.get("limit", 50)),
                cfg=cfg.model_dump(),
            )
            return JoneTestResponse(op=op, message="OK", table=_df_to_table(df)).model_dump()

        if op == JoneOp.neighbors_subgraph:
            v_df, e_df = neighbors_subgraph(
                cc,
                start_vertex=str(params["start_vertex"]),
                lower_bound=int(params.get("lower_bound", 1)),
                upper_bound=int(params.get("upper_bound", 1)),
                direction=str(params.get("direction", "OUTGOING")),
                cfg=cfg.model_dump(),
            )
            graph = JoneGraphResult(nodes=_df_to_table(v_df), edges=_df_to_table(e_df))
            return JoneTestResponse(op=op, message="OK", graph=graph).model_dump()

        if op == JoneOp.shortest_path:
            sp = shortest_path(
                cc,
                source=str(params["source"]),
                target=str(params["target"]),
                direction=str(params.get("direction", "OUTGOING")),
                weight_col=params.get("weight_col"),
                cfg=cfg.model_dump(),
            )
            # sp is dict: {weight, vertices, edges}
            out = {
                "weight": sp.get("weight"),
                "vertices": _df_to_table(sp["vertices"]).model_dump() if sp.get("vertices") is not None else None,
                "edges": _df_to_table(sp["edges"]).model_dump() if sp.get("edges") is not None else None,
            }
            return JoneTestResponse(op=op, message="OK", result=out).model_dump()

        if op == JoneOp.fe_personnel_by_competency:
            df = fe_personnel_by_competency(
                cc,
                competency_query=str(params.get("competency_query", "")),
                fe_id=params.get("fe_id"),
                min_rating=float(params.get("min_rating", 0.0)),
                limit=int(params.get("limit", 50)),
            )
            return JoneTestResponse(op=op, message="OK", table=_df_to_table(df)).model_dump()

        if op == JoneOp.fe_readiness:
            df = fe_readiness(
                cc,
                fe_id=params.get("fe_id"),
                min_fill_rate=float(params.get("min_fill_rate")) if params.get("min_fill_rate") else None,
                limit=int(params.get("limit", 50)),
            )
            return JoneTestResponse(op=op, message="OK", table=_df_to_table(df)).model_dump()

        return JoneTestResponse(ok=False, op=op, message=f"Unknown op: {op.value}").model_dump()

    except Exception as exc:
        logger.exception("[agent_jone] test_call failed: %r", exc)
        return JoneTestResponse(
            ok=False,
            op=op,
            message="Deterministic call failed",
            debug={"error": repr(exc), "op": op.value, "params": params},
        ).model_dump()

    finally:
        try:
            cc.close()
        except Exception:
            pass


# Convenience endpoints (optional but nice)
@router.get("/node/{node_id}")
async def node_lookup(node_id: str):
    cc = get_hanaml_connection(autocommit=True)
    try:
        cfg = JoneAgentConfig()
        obj = get_node(cc, node_id=node_id, cfg=cfg.model_dump())
        return JSONResponse(content={"ok": True, "node": obj})
    finally:
        try:
            cc.close()
        except Exception:
            pass


@router.get("/skill/{skill_node_id}/tasks")
async def tasks(skill_node_id: str, limit: int = 50):
    cc = get_hanaml_connection(autocommit=True)
    try:
        cfg = JoneAgentConfig()
        df = tasks_for_skill(cc, skill_node_id=skill_node_id, limit=limit, cfg=cfg.model_dump())
        return JSONResponse(content={"ok": True, **_df_to_table(df).model_dump()})
    finally:
        try:
            cc.close()
        except Exception:
            pass
