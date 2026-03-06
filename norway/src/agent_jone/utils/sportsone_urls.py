# src/agent_jone/utils/sportsone_urls.py
"""
Sports One deep-link URL generation for analytics drill-down.
Adds DRILL_DOWN_URL column to analytics results containing personnel data.
"""
from typing import Any, Dict, List, Optional
from loguru import logger
from commons.core.config import get_settings

# In-memory cache for Sports One person ID mappings
_sportsone_id_cache: Dict[str, Optional[str]] = {}


def get_sportsone_person_id(internal_person_id: str) -> Optional[str]:
    """
    Look up Sports One person ID from internal person ID.
    Results are cached to avoid repeated database lookups.
    """
    if internal_person_id in _sportsone_id_cache:
        return _sportsone_id_cache[internal_person_id]
    
    sportsone_id = None
    try:
        from commons.hana.hana_conn import get_hana_connection
        conn = get_hana_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT SPORTSONE_PERSON_ID 
            FROM "DFS_HR"."SPORTSONE_PERSON_MAPPING" 
            WHERE PERSON_ID_EXTERNAL = ?
        """, (internal_person_id,))
        row = cursor.fetchone()
        if row:
            sportsone_id = row[0]
        cursor.close()
    except Exception as e:
        logger.debug(f"[sportsone] Could not look up sportsone_id for {internal_person_id}: {e}")
    
    _sportsone_id_cache[internal_person_id] = sportsone_id
    return sportsone_id


def clear_sportsone_id_cache():
    """Clear the in-memory cache for Sports One person ID mappings."""
    global _sportsone_id_cache
    _sportsone_id_cache = {}


def get_player_url(person_id: str, sportsone_person_id: Optional[str] = None) -> Optional[str]:
    """
    Generate Sports One player drill-down URL.
    Returns None if no mapping exists.
    """
    s1_id = sportsone_person_id or get_sportsone_person_id(person_id)
    
    if not s1_id:
        logger.debug(f"[sportsone] No Sports One mapping for person_id={person_id}")
        return None
    
    settings = get_settings()
    base = settings.SPORTS_ONE_BASE
    path = settings.SPORTS_ONE_PLAYER_PATH
    
    url = f"{base}{path}?person_id={s1_id}"
    return url


def get_team_url(team_id: str, club_id: Optional[str] = None) -> str:
    """Generate Sports One team drill-down URL."""
    settings = get_settings()
    base = settings.SPORTS_ONE_BASE
    path = settings.SPORTS_ONE_TEAM_PATH
    _club_id = club_id or settings.SPORTS_ONE_DEFAULT_CLUB_ID
    
    return f"{base}{path}?team_id={team_id}&club_id={_club_id}"


def get_club_url(club_id: Optional[str] = None) -> str:
    """Generate Sports One club drill-down URL."""
    settings = get_settings()
    base = settings.SPORTS_ONE_BASE
    path = settings.SPORTS_ONE_CLUB_PATH
    _club_id = club_id or settings.SPORTS_ONE_DEFAULT_CLUB_ID
    
    return f"{base}{path}?club_id={_club_id}"


# Column names that indicate person/player IDs for drill-down
PERSON_ID_COLUMNS = frozenset({
    "USER_ID", "PERSON_ID", "PERSON_ID_EXTERNAL",
    "INCUMBENT_USER_ID", "person_id", "user_id",
})

# Column names that indicate team/unit IDs for drill-down
TEAM_ID_COLUMNS = frozenset({
    "FE_ID", "NODE_ID", "UNIT_ID", "TEAM_ID",
    "fe_id", "node_id",
})


def inject_drill_down_urls(
    rows: List[Dict[str, Any]],
    columns: List[str],
    *,
    entity_type: str = "auto",
) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    Inject DRILL_DOWN_URL column into analytics rows.
    
    Args:
        rows: List of row dicts from analytics
        columns: List of column names
        entity_type: "player", "team", "club", or "auto" to detect
    """
    if not rows:
        return rows, columns
    
    # Detect entity type based on columns if auto
    if entity_type == "auto":
        row_cols = set(columns)
        
        for col in PERSON_ID_COLUMNS:
            if col in row_cols:
                entity_type = "player"
                break
        
        if entity_type == "auto":
            for col in TEAM_ID_COLUMNS:
                if col in row_cols:
                    entity_type = "team"
                    break
        
        if entity_type == "auto":
            logger.debug("[sportsone] No ID column found for drill-down URLs")
            return rows, columns
    
    # Inject URLs
    updated_rows = []
    for row in rows:
        new_row = dict(row)
        
        if entity_type == "player":
            pid = None
            for col in PERSON_ID_COLUMNS:
                if col in row and row[col]:
                    pid = str(row[col])
                    break
            if pid:
                url = get_player_url(pid)
                if url:
                    new_row["DRILL_DOWN_URL"] = url
        
        elif entity_type == "team":
            tid = None
            for col in TEAM_ID_COLUMNS:
                if col in row and row[col]:
                    tid = str(row[col])
                    break
            if tid:
                new_row["DRILL_DOWN_URL"] = get_team_url(tid)
        
        elif entity_type == "club":
            new_row["DRILL_DOWN_URL"] = get_club_url()
        
        updated_rows.append(new_row)
    
    # Add URL column to columns list
    if any("DRILL_DOWN_URL" in r for r in updated_rows):
        updated_columns = columns + ["DRILL_DOWN_URL"]
    else:
        updated_columns = columns
    
    return updated_rows, updated_columns


def inject_drill_down_urls_to_analytics(
    analytics: Dict[str, Any],
    *,
    entity_type: str = "auto",
) -> Dict[str, Any]:
    """
    Inject DRILL_DOWN_URL into an analytics spec dict.
    """
    if not analytics or analytics.get("type") != "table":
        return analytics
    
    rows = analytics.get("rows", [])
    columns = analytics.get("columns", [])
    
    updated_rows, updated_columns = inject_drill_down_urls(
        rows, columns, entity_type=entity_type
    )
    
    return {
        **analytics,
        "columns": updated_columns,
        "rows": updated_rows,
    }
