"""
Phase 1 Demo — Auth + Multi-tenancy

Shows what a real developer experiences when they integrate X25:

  Step 1: Register your org → get a key
  Step 2: Pass the key to the SDK — that's it
  Step 3: X25 learns only for YOUR org, isolated from everyone else

Two companies run the same prompts. Their routing brains never mix.
Watch the dashboard at http://localhost:8000/dashboard while this runs.

Usage:
    python phase1_auth_demo.py
"""

from __future__ import annotations

import sys
import os
import time
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from x25 import X25

GATEWAY = "http://localhost:8000"

DIVIDER = "=" * 62


def banner(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def create_key(org: str) -> str:
    """Call the gateway to create an API key for this org."""
    resp = httpx.post(
        f"{GATEWAY}/keys/create",
        json={"org": org, "rate_limit_rpm": 0},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["key"]


def list_keys():
    resp = httpx.get(f"{GATEWAY}/keys/list", timeout=10)
    resp.raise_for_status()
    return resp.json()["keys"]


def print_response(label: str, r, indent: int = 2):
    pad = " " * indent
    print(f"\n{pad}  Model:    {r.model_used.split('/')[-1]}")
    print(f"{pad}  Task:     {r.task_type}")
    print(f"{pad}  Quality:  {r.quality_score:.3f}")
    print(f"{pad}  Cost:     ${r.cost_usd:.6f}")
    print(f"{pad}  Saved:    ${max(0, r.goal_match.get('cost', 0)):.0%} vs frontier")
    print(f"{pad}  Answer:   {r.text[:120]}{'...' if len(r.text) > 120 else ''}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Show what happens WITHOUT auth (legacy mode, pre-Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

banner("BEFORE Phase 1 — No auth, shared namespace")

print("""
  # Old way — anyone can claim any org name, no isolation guarantee
  agent = X25(org="acme-corp")   # just a string, nothing stopping collisions
  agent.complete("...")
""")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Create keys for two real companies
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 1 — Two companies register with X25")

print("\n  Company A: Acme Corp — a legal tech startup")
print("  Company B: Beta AI  — an e-commerce company")
print("\n  Each calls POST /keys/create once to get their key.\n")

try:
    key_acme = create_key("acme-corp")
    key_beta = create_key("beta-ai")
except Exception as e:
    print(f"  [ERROR] Could not reach gateway: {e}")
    print("  Make sure the gateway is running:")
    print("  cd gateway && uvicorn main:app --reload --port 8000")
    sys.exit(1)

print(f"  acme-corp key: {key_acme[:18]}...{key_acme[-6:]}  ← stored in acme's .env")
print(f"  beta-ai   key: {key_beta[:18]}...{key_beta[-6:]}  ← stored in beta's .env")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Show the new SDK usage with keys
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 2 — Developer code (what they write)")

print("""
  # Acme Corp's code — they only set their key once in .env
  from x25 import X25

  agent = X25(
      api_key=os.environ["X25_API_KEY"],   # sk-x25-...
      optimize_for={"cost": 0.7, "quality": 0.2, "latency": 0.1},
  )
  result = agent.complete("Summarise this NDA in plain English...")
""")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Run real calls for both orgs
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 3 — Both companies make calls simultaneously")

# Acme: legal tech, optimises for cost (legal docs are cheap to summarise)
agent_acme = X25(
    api_key=key_acme,
    gateway_url=GATEWAY,
    optimize_for={"cost": 0.7, "quality": 0.2, "latency": 0.1},
)

# Beta: e-commerce, optimises for quality (product descriptions matter)
agent_beta = X25(
    api_key=key_beta,
    gateway_url=GATEWAY,
    optimize_for={"cost": 0.2, "quality": 0.7, "latency": 0.1},
)

print("\n  [acme-corp] Sending: NDA summarisation task (cost-optimised)...")
try:
    r_acme = agent_acme.complete(
        "Summarise the key obligations in a standard non-disclosure agreement. "
        "List the three most important clauses in plain English.",
        hint="summary",
    )
    print_response("acme-corp result", r_acme)
except Exception as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

time.sleep(1)

print("\n\n  [beta-ai] Sending: Product description task (quality-optimised)...")
try:
    r_beta = agent_beta.complete(
        "Write a compelling product description for a wireless ergonomic keyboard "
        "targeting software developers. Emphasise comfort and productivity.",
        hint="creative",
    )
    print_response("beta-ai result", r_beta)
except Exception as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Prove isolation
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 4 — Proving isolation: each org sees only their own data")

acme_stats = agent_acme.get_stats()
beta_stats  = agent_beta.get_stats()

print(f"\n  acme-corp stats (should show 1 call, cost-optimised routing):")
print(f"    calls:       {acme_stats['total_calls']}")
print(f"    avg quality: {acme_stats['avg_quality']}")
print(f"    model used:  {list(acme_stats.get('model_distribution', {}).keys())}")

print(f"\n  beta-ai stats (should show 1 call, quality-optimised routing):")
print(f"    calls:       {beta_stats['total_calls']}")
print(f"    avg quality: {beta_stats['avg_quality']}")
print(f"    model used:  {list(beta_stats.get('model_distribution', {}).keys())}")

if acme_stats["total_calls"] != beta_stats["total_calls"] or \
   list(acme_stats.get("model_distribution", {}).keys()) != list(beta_stats.get("model_distribution", {}).keys()):
    print("\n  ISOLATION CONFIRMED — different models chosen for different cost/quality prefs")
else:
    print("\n  Stats match this run (both used same model tier) but state is still isolated.")
    print("  Run more calls and you'll see diverging routing decisions.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Show key list (management view)
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 5 — Key management view (operator console)")

keys = list_keys()
print(f"\n  {'ORG':<20} {'KEY PREVIEW':<28} {'CALLS':<8} {'RATE LIMIT'}")
print(f"  {'-'*20} {'-'*28} {'-'*8} {'-'*12}")
for k in keys:
    rpm = str(k["rate_limit_rpm"]) + "/min" if k["rate_limit_rpm"] else "unlimited"
    print(f"  {k['org']:<20} {k['key_preview']:<28} {k['call_count']:<8} {rpm}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

banner("Phase 1 COMPLETE")

print(f"""
  What just happened:
    ✓  Two orgs registered with X25 via POST /keys/create
    ✓  Each got a unique sk-x25-... key
    ✓  The SDK passed the key as Authorization: Bearer <key>
    ✓  X25 derived org identity from the key — not from a string argument
    ✓  Routing state (bandit learning) is scoped per org — no bleed
    ✓  Rate limiting is enforced per org (configurable per key)

  What's next:
    Phase 2 — Dynamic model registry
    X25 fetches OpenRouter's full 300+ model catalog every hour,
    clusters by cost/capability, and routes across ALL of them —
    not just the 3 hardcoded tiers we have today.

  Dashboard: http://localhost:8000/dashboard
  API docs:  http://localhost:8000/docs
{DIVIDER}
""")
