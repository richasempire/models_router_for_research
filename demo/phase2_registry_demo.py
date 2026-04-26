"""
Phase 2 Demo — Dynamic Model Registry

Shows the difference between Phase 1 (3 hardcoded models) and
Phase 2 (live catalog of 300+ models, auto-clustered into tiers).

What you'll see:
  1. X25 fetches the OpenRouter catalog on startup
  2. 300+ models auto-clustered into free / slm / mid / frontier / vlm
  3. The best model per tier is chosen dynamically (context window + recency)
  4. A real routing call — same SDK, but now picks from the live catalog
  5. What happens if a new model releases tomorrow: zero code changes

Run this while the gateway is running at http://localhost:8000

Usage:
    python phase2_registry_demo.py
"""

from __future__ import annotations

import sys
import os
import httpx
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from x25 import X25

GATEWAY = "http://localhost:8000"
DIVIDER = "=" * 62


def banner(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def get_registry():
    resp = httpx.get(f"{GATEWAY}/registry", timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_all_models():
    resp = httpx.get(f"{GATEWAY}/registry/all", timeout=15)
    resp.raise_for_status()
    return resp.json()


def create_key(org: str) -> str:
    resp = httpx.post(f"{GATEWAY}/keys/create", json={"org": org}, timeout=10)
    resp.raise_for_status()
    return resp.json()["key"]


def print_response(r, indent=2):
    pad = " " * indent
    print(f"{pad}  Model:    {r.model_used.split('/')[-1]}  [{r.model_used.split('/')[0]}]")
    print(f"{pad}  Task:     {r.task_type}")
    print(f"{pad}  Quality:  {r.quality_score:.3f}")
    print(f"{pad}  Cost:     ${r.cost_usd:.6f}")
    print(f"{pad}  Latency:  {r.latency_ms:.0f}ms")
    print(f"{pad}  Answer:   {r.text[:140]}{'...' if len(r.text) > 140 else ''}")


# ─────────────────────────────────────────────────────────────────────────────
# BEFORE: Phase 1 — hardcoded
# ─────────────────────────────────────────────────────────────────────────────

banner("BEFORE Phase 2 — Hardcoded to 3 models")

print("""
  CASCADE_TIERS = [
      {"tier": "slm",      "model": "openai/gpt-4o-mini"},        # hardcoded
      {"tier": "mid",      "model": "anthropic/claude-haiku-4-5"}, # hardcoded
      {"tier": "frontier", "model": "anthropic/claude-sonnet-4-6"} # hardcoded
  ]

  Problems:
    - New model releases → you have to edit code and redeploy
    - Free/open-source models ignored entirely
    - Vision models not routable
    - Pricing goes stale the moment OpenRouter updates it
""")

# ─────────────────────────────────────────────────────────────────────────────
# AFTER: Phase 2 — live catalog
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 1 — X25 fetches live catalog from OpenRouter")

try:
    catalog = get_registry()
except Exception as e:
    print(f"\n  [ERROR] Could not reach gateway: {e}")
    print("  Start with: cd gateway && uvicorn main:app --reload --port 8000")
    sys.exit(1)

print(f"\n  Total models discovered: {catalog['total_models']}")
print(f"  Last refreshed: {catalog['last_updated']:.0f} (unix)")
print()
print(f"  {'TIER':<12} {'BEST MODEL':<50} {'COST/1M OUTPUT':<16} {'CANDIDATES'}")
print(f"  {'-'*12} {'-'*50} {'-'*16} {'-'*10}")
for tier in catalog["tiers"]:
    cost = f"${tier['cost_per_1m_output']:.3f}" if tier['cost_per_1m_output'] > 0 else "FREE"
    print(f"  {tier['tier']:<12} {tier['best_model']:<50} {cost:<16} {tier['candidates']}")

# ─────────────────────────────────────────────────────────────────────────────
# Show distribution across tiers
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 2 — Model distribution across tiers")

all_data = get_all_models()
all_models = all_data["models"]

by_tier: dict = {}
for m in all_models:
    by_tier.setdefault(m["tier"], []).append(m)

for tier_name in ["free", "slm", "mid", "frontier", "vlm"]:
    members = by_tier.get(tier_name, [])
    if not members:
        continue
    examples = [m["id"].split("/")[-1] for m in members[:4]]
    print(f"\n  {tier_name.upper():<12} ({len(members)} models)")
    print(f"    examples: {', '.join(examples)}")
    if members:
        costs = [m["cost_per_1m_output"] for m in members if m["cost_per_1m_output"] > 0]
        if costs:
            print(f"    cost range: ${min(costs):.3f} – ${max(costs):.2f} per 1M output tokens")

# ─────────────────────────────────────────────────────────────────────────────
# Real routing call — uses live registry
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 3 — Real routing call (same SDK, dynamic model)")

key = create_key("phase2-demo")
agent = X25(
    api_key=key,
    gateway_url=GATEWAY,
    optimize_for={"cost": 0.6, "quality": 0.3, "latency": 0.1},
)

print(f"\n  Sending a reasoning task...")
print(f"  X25 will pick the cheapest model that can handle it.\n")

try:
    r = agent.complete(
        "Explain the difference between a transformer and an RNN in 3 sentences.",
        hint="reasoning",
    )
    print_response(r)
except Exception as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Show what "zero code change on new model release" looks like
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 4 — What happens when a new model releases tomorrow")

print("""
  Before Phase 2:
    1. New model releases on OpenRouter
    2. You manually find out about it
    3. You update CASCADE_TIERS in openrouter.py
    4. You redeploy the gateway
    5. You test it works

  After Phase 2:
    1. New model releases on OpenRouter
    2. X25 discovers it on next hourly refresh (or restart)
    3. If it scores better than current best for its tier → auto-promoted
    4. Done. Zero code changes.

  The registry refresh runs every 3600 seconds in the background.
  Force a refresh by restarting the gateway.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

banner("Phase 2 COMPLETE")

print(f"""
  What changed:
    ✓  CASCADE_TIERS removed from openrouter.py — no hardcoded models
    ✓  ModelRegistry fetches OpenRouter /v1/models every hour
    ✓  {catalog['total_models']} models auto-clustered into free/slm/mid/frontier/vlm
    ✓  Best model per tier chosen by context window + recency score
    ✓  agent.py resolves tier → model at call time, not at deploy time
    ✓  New endpoints: GET /registry, GET /registry/all

  What's next:
    Phase 3 — Thompson Sampling
    Instead of LinUCB across 3 fixed tiers, Thompson Sampling learns
    which tier works best for YOUR specific workload — with proper
    Bayesian uncertainty so it explores intelligently from call #1.

  Dashboard: http://localhost:8000/dashboard
  Registry:  http://localhost:8000/registry
  All models: http://localhost:8000/registry/all
{DIVIDER}
""")
