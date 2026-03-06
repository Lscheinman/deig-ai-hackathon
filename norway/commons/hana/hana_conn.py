# commons/hana/hana_conn.py
"""
HANA connection utilities for JONE agent.
Provides both raw hdbcli and hana_ml ConnectionContext connections.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional, Any

from hdbcli import dbapi
from loguru import logger as log

from commons.core.config import get_settings

try:
    from hana_ml.dataframe import ConnectionContext as HanaMLConnectionContext
except ImportError:
    HanaMLConnectionContext = None


def _unwrap_connection(conn_or_ctx: Any):
    """Normalize connection to something cursor/commit/rollback compatible."""
    if isinstance(conn_or_ctx, dbapi.Connection):
        return conn_or_ctx

    if HanaMLConnectionContext is not None and isinstance(conn_or_ctx, HanaMLConnectionContext):
        underlying = getattr(conn_or_ctx, "connection", None)
        if underlying is None:
            raise TypeError("hana_ml ConnectionContext has no underlying .connection")
        return underlying

    required = ("cursor", "commit", "rollback", "close")
    if all(hasattr(conn_or_ctx, name) for name in required):
        return conn_or_ctx

    raise TypeError(f"Unsupported connection type {type(conn_or_ctx)!r}")


def get_hana_connection(autocommit: Optional[bool] = None) -> dbapi.Connection:
    """Get a raw hdbcli HANA connection."""
    s = get_settings()
    if autocommit is None:
        autocommit = True

    log.bind(host=s.HANA_HOST, port=s.HANA_PORT).info("HANA: connecting")
    conn = dbapi.connect(
        address=s.HANA_HOST,
        port=int(s.HANA_PORT),
        user=s.HANA_USER,
        password=s.HANA_PASSWORD,
        encrypt=getattr(s, "HANA_ENCRYPT", True),
        sslValidateCertificate=getattr(s, "HANA_SSL_VALIDATE", False),
    )
    conn.setautocommit(autocommit)
    log.info("HANA: connected (autocommit={})", autocommit)
    return conn


@contextmanager
def ctx_hana(autocommit: bool = False) -> Iterator[dbapi.Connection]:
    """Context manager for HANA connection with auto-commit/rollback."""
    conn = get_hana_connection(autocommit=autocommit)
    try:
        yield conn
        if not autocommit:
            conn.commit()
    except Exception:
        if not autocommit:
            try:
                conn.rollback()
            except Exception:
                log.warning("HANA: rollback failed")
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_hanaml_connection(autocommit: Optional[bool] = None) -> "HanaMLConnectionContext":
    """Create a hana_ml ConnectionContext."""
    if HanaMLConnectionContext is None:
        raise ImportError("hana-ml is not installed. Install with `pip install hana-ml`.")

    s = get_settings()
    if autocommit is None:
        autocommit = True

    log.bind(host=s.HANA_HOST, port=s.HANA_PORT).info("HANA-ML: connecting")
    try:
        cc = HanaMLConnectionContext(
            address=s.HANA_HOST,
            port=int(s.HANA_PORT),
            user=s.HANA_USER,
            password=s.HANA_PASSWORD,
            encrypt=getattr(s, "HANA_ENCRYPT", True),
            ssl_validate_certificate=getattr(s, "HANA_SSL_VALIDATE", False),
        )

        try:
            raw = _unwrap_connection(cc)
            raw.setautocommit(autocommit)
        except Exception:
            log.warning("HANA-ML: unable to set autocommit; continuing")

        log.info("HANA-ML: connected (autocommit={})", autocommit)
        return cc
    except Exception as e:
        log.bind(host=s.HANA_HOST, port=s.HANA_PORT, err=str(e)).error("HANA-ML: connect failed")
        raise


@contextmanager
def hanaml_conn(autocommit: Optional[bool] = None) -> Iterator["HanaMLConnectionContext"]:
    """Context manager for hana_ml ConnectionContext."""
    cc = get_hanaml_connection(autocommit=autocommit)
    try:
        yield cc
    finally:
        try:
            cc.close()
        except Exception:
            pass
