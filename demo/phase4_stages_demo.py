"""
Phase 4 Demo — Stage System

Shows X25 automatically advancing an org through improvement stages
as calls accumulate. Watch the stage banner update on the dashboard.

Stages:
  1 — Explore    (0–49 calls)   Learning your task mix
  2 — Exploit    (50–199 calls) Routing converged, personalised
  3 — Feedback   (200–499 calls) Provide examples to improve
  4 — Fine-tune  (500+ calls)   SLM fine-tuning unlocked

Usage:
    python phase4_stages_demo.py
"""

from __future__ import annotations

import sys
import os
import time
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from x25 import X25

GATEWAY  = "http://localhost:8000"
DIVIDER  = "=" * 62


def banner(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def create_key(org: str) -> str:
    resp = httpx.post(f"{GATEWAY}/keys/create", json={"org": org}, timeout=10)
    resp.raise_for_status()
    return resp.json()["key"]


def get_stage(org: str) -> dict:
    resp = httpx.get(f"{GATEWAY}/stage/{org}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def submit_feedback(org: str, key: str, prompt: str, model: str) -> dict:
    resp = httpx.post(
        f"{GATEWAY}/feedback/{org}",
        json={"prompt": prompt, "good_model": model},
        headers={"Authorization": f"Bearer {key}"},
        timeout=10,
    )
    return resp.json() if resp.status_code == 200 else {"error": resp.text}


def print_stage(s: dict):
    unlocked = [st for st in s["all_stages"] if st["unlocked"]]
    locked   = [st for st in s["all_stages"] if not st["unlocked"]]
    bar_len  = int(s["progress_in_stage"] * 30)
    bar      = "█" * bar_len + "░" * (30 - bar_len)

    print(f"\n  Stage {s['stage']} — {s['stage_name']}  [{bar}]  {s['progress_in_stage']*100:.0f}%")
    print(f"  {s['stage_description']}")
    print(f"  Total calls: {s['total_calls']}")
    if s["calls_to_next_stage"]:
        print(f"  Next: Stage {s['next_stage']} ({s['next_stage_name']}) in {s['calls_to_next_stage']} calls")
    else:
        print(f"  Fine-tune ready — POST /improve to start")
    print(f"  Unlocked: {[st['name'] for st in unlocked]}  |  Locked: {[st['name'] for st in locked]}")
    if s.get("drift_detected"):
        print(f"  ⚠️  DRIFT: Quality dropped from baseline — re-routing recommended")
    if s.get("improvement_available"):
        print(f"  ✓  Improvement available — submit examples via POST /feedback")


# ─────────────────────────────────────────────────────────────────────────────
banner("Phase 4 — Stage System Demo")

print("""
  X25 automatically tracks how many calls each org has made
  and advances them through 4 improvement stages.

  No cron jobs. No manual triggers.
  X25 watches every routing decision and advances stages autonomously.
""")

try:
    key = create_key("stage-demo-org")
except Exception as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

agent = X25(
    api_key=key,
    gateway_url=GATEWAY,
    optimize_for={"cost": 0.5, "quality": 0.4, "latency": 0.1},
)

print(f"  Org: {agent.org}")

# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 1 — Initial state (Stage 1: Explore)")

s = get_stage(agent.org)
print_stage(s)

# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 2 — Making calls and watching stage advance")

# We'll use a fast batch of simple prompts to accumulate calls quickly
batch_prompts = [
    ("Classify: is 'buy now!' spam?",                         "classification"),
    ("Summarise: DNA carries genetic information.",            "summary"),
    ("Extract numbers from: call 555-1234 or 555-5678",       "extraction"),
    ("Is 'free money' an urgent email?",                       "classification"),
    ("Summarise: photosynthesis converts light to energy.",    "summary"),
]

print(f"\n  Sending {len(batch_prompts)} calls...\n")

stage_before = s["stage"]
for i, (prompt, hint) in enumerate(batch_prompts):
    try:
        r = agent.complete(prompt, hint=hint)
        s = get_stage(agent.org)
        if s["stage"] != stage_before:
            print(f"\n  *** STAGE ADVANCED: {stage_before} → {s['stage']} ({s['stage_name']}) ***")
            print_stage(s)
            stage_before = s["stage"]
        else:
            print(f"  call {i+1:2d}: {hint:16s} → {r.model_used.split('/')[-1]:25s}  "
                  f"q={r.quality_score:.2f}  stage={s['stage']}  "
                  f"({s['total_calls']}/{50 if s['stage']==1 else 200} calls)")
    except Exception as e:
        print(f"  call {i+1}: [ERROR] {e}")

# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 3 — Final stage state")

s = get_stage(agent.org)
print_stage(s)

# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 4 — Stage 3 preview: submitting feedback examples")

print(f"""
  At Stage 3 (200+ calls), orgs can submit labelled examples:
    "For this type of prompt, model X gave the best answer."

  X25 stores these and uses them in Phase 5 to fine-tune a
  lightweight SLM on your specific task patterns.

  Example SDK call:
    POST /feedback/{{org}}
    {{
      "prompt":     "Summarise this legal contract...",
      "good_model": "anthropic/claude-sonnet-4-6"
    }}
""")

# Try to submit feedback (will be blocked if stage < 3, which shows the gate)
result = submit_feedback(
    org=agent.org,
    key=key,
    prompt="Summarise the key clauses in this NDA.",
    model="anthropic/claude-sonnet-4-6",
)

if "error" in result:
    print(f"  Feedback blocked (expected at Stage {s['stage']}): {result['error'][:120]}")
    print(f"  This gate opens at Stage 3 (200 calls). Currently at {s['total_calls']} calls.")
else:
    print(f"  Feedback accepted: {result}")

# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 5 — Operator view: all orgs and their stages")

resp = httpx.get(f"{GATEWAY}/stages/all", timeout=10)
all_orgs = resp.json().get("orgs", [])
print(f"\n  {'ORG':<25} {'STAGE':<8} {'CALLS':<8} {'IMPROVE':<10} {'DRIFT'}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*6}")
for o in all_orgs[:10]:
    improve = "✓ ready" if o["improvement_available"] else "—"
    drift   = "⚠️ yes" if o["drift_detected"] else "no"
    print(f"  {o['org']:<25} {o['stage']:<8} {o['total_calls']:<8} {improve:<10} {drift}")

# ─────────────────────────────────────────────────────────────────────────────
banner("Phase 4 COMPLETE")

print(f"""
  What changed:
    ✓  gateway/stages.py: StageTracker, 4 stages, drift monitor
    ✓  agent.py: record_call() fires after every routing decision
    ✓  GET /stage/{{org}}: full stage status with progress bar data
    ✓  GET /stages/all: operator view of all orgs
    ✓  POST /feedback/{{org}}: labelled examples (Stage 3+ gated)
    ✓  Dashboard: stage banner with progress bar and step indicators
    ✓  Weekly drift monitor: flags orgs whose quality dropped >10%

  What's next:
    Phase 5 — Fine-tuning Pipeline
    When an org hits Stage 4 (500+ calls), X25 takes their logged
    call history, formats it as LoRA training data, and fine-tunes
    a Llama 3.2 3B model on their exact task patterns via Unsloth.
    That model re-enters the routing pool as a custom SLM.

  Dashboard:    http://localhost:8000/dashboard
  Stage status: http://localhost:8000/stage/{agent.org}
  All stages:   http://localhost:8000/stages/all
{DIVIDER}
""")
