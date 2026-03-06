#!/usr/bin/env python3
# run.py - JONE Agent standalone runner
"""
JONE (JOb Norsk Engine) - HR Graph Agent

A deterministic HR graph assistant for Norwegian military force readiness.
Uses SAP HANA Graph for skills, tasks, qualities, and personnel data.

Usage:
    python run.py
    
Or with uvicorn directly:
    uvicorn run:app --host 0.0.0.0 --port 8080 --reload
"""
import sys
from pathlib import Path

# Add src and root to path for imports
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from agent_jone.api import router as jone_router

# Create FastAPI app
app = FastAPI(
    title="JONE Agent",
    description="Norwegian HR Graph Agent - Skills, Tasks, Qualities & Personnel",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the JONE router
app.include_router(jone_router, prefix="/jone")


@app.get("/")
async def root():
    return {
        "agent": "JONE",
        "name": "JOb Norsk Engine",
        "description": "Norwegian HR Graph Agent for military force readiness",
        "endpoints": {
            "chat": "/jone/agent/jone/chat",
            "stream": "/jone/agent/jone/stream",
            "health": "/jone/agent/jone/health",
            "test": "/jone/agent/jone/test",
        }
    }


@app.get("/health")
async def health():
    return {"ok": True, "agent": "jone"}


if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting JONE Agent on {host}:{port}")
    uvicorn.run(
        "run:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )
