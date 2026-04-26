"""
X25 Gateway — FastAPI server.

Three surfaces:
  POST /route          ← the SDK calls this
  GET  /stats/{org}    ← dashboard fetches aggregated stats
  GET  /audit/{org}    ← dashboard fetches recent audit records
  GET  /verify         ← prove the audit chain is intact
  WS   /ws             ← live event stream to dashboard
  MCP  /mcp            ← X25 as a tool in Claude Code / Cursor
"""

from __future__ import annotations

import sys
import os
import json
import asyncio
from typing import Optional

# Make gateway modules importable
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from agent import run_routing
from audit import AuditStore

load_dotenv()

app = FastAPI(title="X25 Routing Agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_audit = AuditStore()

# ── WebSocket connection manager ───────────────────────────────────────────────

class ConnectionManager:
    """Manages live WebSocket connections to the dashboard."""

    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


def make_broadcast_fn(loop: asyncio.AbstractEventLoop):
    """Create a sync broadcast function that works from sync routing code."""
    def broadcast_fn(message: dict):
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(manager.broadcast(message), loop)
    return broadcast_fn


# ── Request / Response schemas ─────────────────────────────────────────────────

class RouteRequest(BaseModel):
    prompt: str
    org: str = "default"
    optimize_for: dict = {"cost": 0.33, "quality": 0.34, "latency": 0.33}
    policy: dict = {}
    hint: Optional[str] = None


class RouteResponse(BaseModel):
    text: str
    model_used: str
    provider: str
    task_type: str
    cost_usd: float
    latency_ms: float
    quality_score: float
    cascade_steps: int
    audit_hash: str
    goal_match: dict


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post("/route", response_model=RouteResponse)
async def route(req: RouteRequest):
    """
    Main routing endpoint. Runs the full LangGraph agentic loop:
    classify → select (LinUCB) → dispatch → judge → escalate? → learn → audit
    """
    loop = asyncio.get_event_loop()
    broadcast_fn = make_broadcast_fn(loop)

    # Run the agent in a thread pool (it makes synchronous HTTP calls)
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: run_routing(
            prompt=req.prompt,
            org=req.org,
            optimize_for=req.optimize_for,
            policy=req.policy,
            hint=req.hint,
            broadcast_fn=broadcast_fn,
        ),
    )

    return RouteResponse(
        text=result["response_text"],
        model_used=result["model_used"],
        provider=result["provider"],
        task_type=result["task_type"],
        cost_usd=result["cost_usd"],
        latency_ms=result["latency_ms"],
        quality_score=result["quality_score"],
        cascade_steps=result["cascade_steps"],
        audit_hash=result["audit_hash"],
        goal_match=result.get("goal_match", {}),
    )


@app.get("/stats/{org}")
async def get_stats(org: str):
    """Aggregated routing stats for dashboard panels."""
    return _audit.get_stats(org=org if org != "all" else None)


@app.get("/audit/{org}")
async def get_audit(org: str, limit: int = 20):
    """Recent audit records for the audit trail panel."""
    return {
        "records": _audit.get_recent(
            org=org if org != "all" else None,
            limit=limit,
        )
    }


@app.get("/verify")
async def verify_chain():
    """Verify the hash chain is intact — tamper detection."""
    ok, message = _audit.verify_chain()
    return {"intact": ok, "message": message}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "X25 Routing Agent"}


# ── WebSocket endpoint ─────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Live event stream to dashboard. Each routing decision broadcasts here."""
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()  # keep-alive
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ── MCP Server ─────────────────────────────────────────────────────────────────

try:
    from fastapi_mcp import FastApiMCP

    mcp = FastApiMCP(
        app,
        name="X25 Routing Agent",
        description=(
            "Autonomous LLM routing agent. Call x25_route to send any prompt "
            "and X25 will autonomously select the optimal model from 300+ options, "
            "run a cascade, evaluate quality, and return the result with full audit trail."
        ),
        include_operations=["route"],
    )
    mcp.mount()
    print("✅ MCP server mounted at /mcp — connect from Claude Code or Cursor")
except Exception as e:
    print(f"⚠️  MCP server skipped: {e}")


# ── Dashboard static file ──────────────────────────────────────────────────────

dashboard_path = os.path.join(os.path.dirname(__file__), "..", "dashboard")
if os.path.exists(dashboard_path):
    app.mount("/dashboard", StaticFiles(directory=dashboard_path, html=True), name="dashboard")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
