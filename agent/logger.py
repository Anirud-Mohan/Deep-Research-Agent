"""Session logger — writes one JSON file per research session."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

LOGS_DIR = Path("logs")


def _ensure_dir() -> None:
    LOGS_DIR.mkdir(exist_ok=True)


def save_session(
    *,
    session_id: str,
    query: str,
    sub_queries: list[str],
    findings: list[dict],
    sources: list[dict],
    answer: str,
    memory_state: dict,
    budget: dict,
) -> str:
    """Persist a full research session to ``logs/<session_id>.json``."""
    _ensure_dir()

    record = {
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "sub_queries": sub_queries,
        "findings": findings,
        "sources": sources,
        "answer": answer,
        "memory_state": memory_state,
        "budget": budget,
    }

    path = LOGS_DIR / f"{session_id}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2)

    return str(path)


def list_sessions() -> list[dict]:
    """Return metadata for all saved sessions, newest first."""
    _ensure_dir()
    sessions = []
    for p in sorted(LOGS_DIR.glob("*.json"), key=os.path.getmtime, reverse=True):
        try:
            with open(p) as f:
                data = json.load(f)
            sessions.append({
                "session_id": data["session_id"],
                "timestamp": data["timestamp"],
                "query": data["query"],
                "sub_queries_count": len(data.get("sub_queries", [])),
                "total_tokens": data.get("budget", {}).get("total_tokens", 0),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return sessions


def load_session(session_id: str) -> dict | None:
    """Load a previously saved session by ID."""
    path = LOGS_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
