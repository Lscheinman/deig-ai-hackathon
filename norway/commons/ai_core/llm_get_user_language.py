# commons/ai_core/llm_get_user_language.py
"""
Language detection utilities for JONE agent.
"""
import json
import re

_LANG_RE = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8}){0,3}$")  # loose BCP-47-ish


async def get_user_language_tag(user_query: str) -> str:
    """
    Detect user language from query text.
    Returns BCP-47-ish tag like 'en', 'nb', 'zh-Hans', or 'und' if uncertain.
    """
    # Simple heuristic fallback - check for common language patterns
    q = user_query.lower()
    
    # Norwegian keywords
    if any(kw in q for kw in ["hvem", "hva", "hvor", "hvorfor", "kan", "vil", "har", "med", "som", "til"]):
        return "nb"
    
    # German keywords
    if any(kw in q for kw in ["wer", "was", "wo", "warum", "kann", "will", "hat", "mit", "als"]):
        return "de"
    
    # Chinese characters
    if any('\u4e00' <= c <= '\u9fff' for c in q):
        return "zh"
    
    # Default to English
    return "en"


async def llm_choose_language_tag(*, llm, user_query: str) -> str:
    """
    Use LLM to detect user's language as a BCP-47-ish tag.
    Falls back to 'und' (undetermined) if detection fails.
    """
    system = (
        "Detect the language of the user's question.\n"
        "Return ONLY strict JSON: {\"language_tag\":\"<bcp47>\"}.\n"
        "Use short tags when possible (e.g. 'zh', 'en', 'no', 'nb').\n"
        "If uncertain, use 'und'."
    )
    
    try:
        raw = await llm.generate_text(system=system, user=user_query, max_tokens=50)
        obj = json.loads(raw.strip())
        tag = str(obj.get("language_tag") or "").strip()[:20]
        return tag if (_LANG_RE.match(tag) or tag == "und") else "und"
    except Exception:
        return await get_user_language_tag(user_query)
