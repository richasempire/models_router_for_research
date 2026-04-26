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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from agent import run_routing
from audit import AuditStore
from auth import get_auth_store, extract_key_from_header
from model_registry import get_registry
from thompson import get_thompson
from stages import get_stage_tracker

load_dotenv()

app = FastAPI(title="X25 Routing Agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_audit = AuditStore()
_auth  = get_auth_store()

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

# ── Auth schemas ───────────────────────────────────────────────────────────────

class CreateKeyRequest(BaseModel):
    org: str
    rate_limit_rpm: int = 0   # 0 = unlimited


class CreateKeyResponse(BaseModel):
    key: str                  # full key — shown once, store it safely
    org: str
    message: str


# ── Route schemas ──────────────────────────────────────────────────────────────

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

# ── Key management endpoints ───────────────────────────────────────────────────

@app.post("/keys/create", response_model=CreateKeyResponse)
async def create_key(req: CreateKeyRequest):
    """
    Create an API key for an org.
    The key is shown once — store it in your .env file.
    """
    key = _auth.create_key(org=req.org, rate_limit_rpm=req.rate_limit_rpm)
    return CreateKeyResponse(
        key=key,
        org=req.org,
        message=f"Key created for org '{req.org}'. Store it safely — shown once.",
    )


@app.get("/keys/me")
async def get_my_org(authorization: Optional[str] = Header(default=None)):
    """Resolve the org for the calling API key. Used by the SDK at init."""
    raw_key = extract_key_from_header(authorization)
    if not raw_key:
        raise HTTPException(status_code=401, detail="No API key provided")
    org_key = _auth.validate(raw_key)
    if not org_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return {"org": org_key.org, "call_count": org_key.call_count}


@app.get("/keys/list")
async def list_keys(org: Optional[str] = None):
    """List all keys (previewed, not full) for an org or all orgs."""
    return {"keys": _auth.list_keys(org=org)}


@app.delete("/keys/{key}")
async def revoke_key(key: str):
    """Revoke an API key immediately."""
    revoked = _auth.revoke_key(key)
    if not revoked:
        raise HTTPException(status_code=404, detail="Key not found")
    return {"revoked": True}


# ── Route endpoint ─────────────────────────────────────────────────────────────

@app.post("/route", response_model=RouteResponse)
async def route(req: RouteRequest, authorization: Optional[str] = Header(default=None)):
    """
    Main routing endpoint. Runs the full LangGraph agentic loop:
    classify → select (LinUCB) → dispatch → judge → escalate? → learn → audit

    Auth (optional — backwards compatible):
      Authorization: Bearer sk-x25-<key>
      If a valid key is provided, org is derived from the key.
      If no key, req.org is used (legacy / dev mode).
    """
    # Resolve org from key if provided
    org = req.org
    raw_key = extract_key_from_header(authorization)
    if raw_key:
        org_key = _auth.validate(raw_key)
        if org_key is None:
            raise HTTPException(status_code=401, detail="Invalid API key")
        # Check rate limit
        allowed, count = _auth.check_rate_limit(org_key.org, org_key.rate_limit_per_min)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {count} calls in last 60s (limit: {org_key.rate_limit_per_min})",
            )
        org = org_key.org  # key wins over req.org

    loop = asyncio.get_event_loop()
    broadcast_fn = make_broadcast_fn(loop)

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: run_routing(
            prompt=req.prompt,
            org=org,
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


@app.get("/stage/{org}")
async def stage_status(org: str):
    """Current improvement stage for an org."""
    return get_stage_tracker().get_status(org)


@app.get("/stages/all")
async def all_stages():
    """Operator view — all orgs and their stages."""
    return {"orgs": get_stage_tracker().list_all_orgs()}


class FeedbackRequest(BaseModel):
    prompt: str
    good_model: str


@app.post("/feedback/{org}")
async def submit_feedback(org: str, req: FeedbackRequest,
                          authorization: Optional[str] = Header(default=None)):
    """
    Stage 3+ — submit a labelled example.
    Tells X25 which model produced the best response for a given prompt.
    Used to fine-tune the routing classifier in Phase 5.
    """
    raw_key = extract_key_from_header(authorization)
    if raw_key:
        org_key = _auth.validate(raw_key)
        if org_key:
            org = org_key.org

    tracker = get_stage_tracker()
    state   = tracker.get_status(org)
    if state["stage"] < 3:
        raise HTTPException(
            status_code=403,
            detail=f"Feedback available at Stage 3+. You are at Stage {state['stage']} "
                   f"({state['calls_to_next_stage']} calls until Stage 3).",
        )
    tracker.submit_feedback(org=org, prompt=req.prompt, good_model=req.good_model)
    count = tracker.get_feedback_count(org)
    return {
        "accepted": True,
        "feedback_count": count,
        "message": f"{count} examples stored. Fine-tuning starts at 50 examples (Phase 5).",
    }


@app.get("/thompson/{org}")
async def thompson_state(org: str):
    """
    Current Thompson Sampling state for an org.
    Shows α, β, mean reward, confidence, and uncertainty per tier.
    Useful for understanding how the router has learned so far.
    """
    router = get_thompson(org)
    return {
        "org":       org,
        "algorithm": "thompson_sampling",
        "arms":      router.get_state_summary(),
    }


@app.get("/registry")
async def registry_summary():
    """Live model catalog — tiers, best models, candidate counts."""
    return get_registry().catalog_summary()


@app.get("/registry/all")
async def registry_all():
    """Full model list with tier assignments."""
    models = get_registry().all_models()
    return {
        "total": len(models),
        "models": [
            {
                "id": m.id,
                "name": m.name,
                "tier": m.tier,
                "cost_per_1m_output": m.cost_per_1m_output,
                "context_length": m.context_length,
                "is_vision": m.is_vision,
            }
            for m in sorted(models, key=lambda m: m.cost_per_1m_output)
        ],
    }


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
