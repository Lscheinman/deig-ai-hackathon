# commons/core/config.py
"""
Standalone configuration for JONE agent.
Reads from environment variables and optional .env file.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger as log

# Optional: python-dotenv
try:
    from dotenv import load_dotenv, find_dotenv
except Exception:
    load_dotenv = None
    find_dotenv = None


def _find_env_file() -> Tuple[Optional[str], bool]:
    """Try to locate and load a .env file."""
    envfile = os.getenv("ENV_FILE")
    candidates = []

    if envfile:
        candidates.append(envfile)

    if find_dotenv:
        p = find_dotenv(filename=".env", usecwd=True)
        if p:
            candidates.append(p)

    here = Path(__file__).resolve()
    for up in [here, *here.parents]:
        cand = up.parent / ".env" if up.is_file() else up / ".env"
        candidates.append(str(cand))

    dedup = []
    seen = set()
    for c in candidates:
        c = str(c)
        if c not in seen and Path(c).exists():
            seen.add(c)
            dedup.append(c)

    if dedup and load_dotenv:
        path = dedup[0]
        ok = load_dotenv(dotenv_path=path, override=False)
        return path, bool(ok)

    return (dedup[0] if dedup else None), False


def _get_vcap_services() -> dict:
    raw = os.getenv("VCAP_SERVICES")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _lookup_hana_in_vcap() -> dict:
    v = _get_vcap_services()
    for key in ("hana", "hanatrial", "hdi-shared", "user-provided"):
        entries = v.get(key, [])
        for entry in entries:
            name = entry.get("name", "")
            if "hana" in name.lower():
                c = entry.get("credentials", {})
                return {
                    "HANA_HOST": c.get("host") or c.get("hostname"),
                    "HANA_PORT": c.get("port"),
                    "HANA_USER": c.get("user") or c.get("username"),
                    "HANA_PASSWORD": c.get("password"),
                }
    return {}


def _lookup_aicore_in_vcap() -> dict:
    """Extract AI Core credentials from VCAP_SERVICES."""
    v = _get_vcap_services()
    for key in ("aicore", "user-provided"):
        entries = v.get(key, [])
        for entry in entries:
            name = entry.get("name", "")
            if "ai" in name.lower() and "core" in name.lower():
                c = entry.get("credentials", {})
                return {
                    "AICORE_CLIENT_ID": c.get("clientid") or c.get("client_id"),
                    "AICORE_CLIENT_SECRET": c.get("clientsecret") or c.get("client_secret"),
                    "AICORE_AUTH_URL": c.get("url") or c.get("auth_url"),
                    "AICORE_BASE_URL": c.get("base_url") or c.get("serviceurls", {}).get("AI_API_URL"),
                    "AICORE_RESOURCE_GROUP": c.get("resource_group"),
                }
    return {}


# Merge VCAP credentials into environment early
_vcap_hana = _lookup_hana_in_vcap()
for k, v in _vcap_hana.items():
    if v and not os.getenv(k):
        os.environ[k] = str(v)

_vcap_aicore = _lookup_aicore_in_vcap()
for k, v in _vcap_aicore.items():
    if v and not os.getenv(k):
        os.environ[k] = str(v)


def _as_bool(s: Optional[str], default=False) -> bool:
    if s is None:
        return default
    return str(s).strip().strip('"').strip("'").lower() in {"1", "true", "yes", "y", "on"}


def _clean_str(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return str(s).strip().strip('"').strip("'")


# Load .env ASAP
_DOTENV_PATH, _DOTENV_LOADED = _find_env_file()
if _DOTENV_PATH:
    log.bind(env=_DOTENV_PATH, loaded=_DOTENV_LOADED).info("dotenv: resolved")


@dataclass
class Settings:
    APP_NAME: str = _clean_str(os.getenv("APP_NAME")) or "JONE Agent"
    DEBUG: bool = _as_bool(os.getenv("DEBUG"), False)
    ALLOW_ORIGINS: str = _clean_str(os.getenv("ALLOW_ORIGINS")) or "*"

    # HANA connection
    HANA_HOST: Optional[str] = _clean_str(os.getenv("HANA_HOST"))
    HANA_PORT: Optional[int] = int(os.getenv("HANA_PORT")) if os.getenv("HANA_PORT") else None
    HANA_USER: Optional[str] = _clean_str(os.getenv("HANA_USER"))
    HANA_PASSWORD: Optional[str] = _clean_str(os.getenv("HANA_PASSWORD"))

    # HANA runtime flags
    HANA_ENCRYPT: bool = _as_bool(os.getenv("HANA_ENCRYPT"), True)
    HANA_SSL_VALIDATE: bool = _as_bool(os.getenv("HANA_SSL_VALIDATE"), False)
    HANA_AUTOCOMMIT: bool = _as_bool(os.getenv("HANA_AUTOCOMMIT"), True)
    HANA_SCHEMA: Optional[str] = _clean_str(os.getenv("HANA_SCHEMA"))

    # SAP AI Core
    AICORE_CLIENT_ID: Optional[str] = _clean_str(os.getenv("AICORE_CLIENT_ID"))
    AICORE_CLIENT_SECRET: Optional[str] = _clean_str(os.getenv("AICORE_CLIENT_SECRET"))
    AICORE_AUTH_URL: Optional[str] = _clean_str(os.getenv("AICORE_AUTH_URL"))
    AICORE_BASE_URL: Optional[str] = _clean_str(os.getenv("AICORE_BASE_URL"))
    AICORE_RESOURCE_GROUP: Optional[str] = _clean_str(os.getenv("AICORE_RESOURCE_GROUP"))

    # HR Schema
    HANA_HR_SCHEMA: str = _clean_str(os.getenv("HANA_HR_SCHEMA")) or "DFS_HR"
    HANA_FE_SCHEMA: str = _clean_str(os.getenv("HANA_FE_SCHEMA")) or "DFS_FE"

    # Sports One Drill-Down URLs
    SPORTS_ONE_BASE: str = _clean_str(os.getenv("SPORTS_ONE_BASE")) or "https://s1-test-patch-publicsafety.test.sportsone.cloud.sap/sap/sports/fnd/ui/start/index.html"
    SPORTS_ONE_PLAYER_PATH: str = _clean_str(os.getenv("SPORTS_ONE_PLAYER_PATH")) or "#/content/player//"
    SPORTS_ONE_TEAM_PATH: str = _clean_str(os.getenv("SPORTS_ONE_TEAM_PATH")) or "#/content/ownclub/teams/"
    SPORTS_ONE_CLUB_PATH: str = _clean_str(os.getenv("SPORTS_ONE_CLUB_PATH")) or "#/content/ownclub/teams/"
    SPORTS_ONE_DEFAULT_CLUB_ID: str = _clean_str(os.getenv("SPORTS_ONE_DEFAULT_CLUB_ID")) or "0AABAF10128438A318008BDE3E08B104"

    def with_vcap_overrides(self) -> "Settings":
        ov = _lookup_hana_in_vcap()
        for k, v in ov.items():
            if getattr(self, k, None) in (None, "", 0):
                setattr(self, k, v)
        return self


def get_settings() -> Settings:
    return Settings().with_vcap_overrides()
