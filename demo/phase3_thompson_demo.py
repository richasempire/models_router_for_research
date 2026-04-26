"""
Phase 3 Demo — Thompson Sampling

Shows the difference between LinUCB (Phase 1/2) and Thompson Sampling:

  LinUCB:           deterministic UCB score, requires context vector,
                    needs many calls to converge

  Thompson Sampling: Bayesian — samples from Beta(α,β) distributions,
                    warm-started from model metadata, smart from call #1,
                    shows you confidence levels per tier

Watch the Beta distributions update in real time as calls come in.

Usage:
    python phase3_thompson_demo.py
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
    resp = httpx.post(f"{GATEWAY}/keys/create", json={"org": org}, timeout=10)
    resp.raise_for_status()
    return resp.json()["key"]


def get_thompson_state(org: str) -> dict:
    resp = httpx.get(f"{GATEWAY}/thompson/{org}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def print_thompson_state(state: dict):
    arms = state["arms"]
    print(f"\n  {'TIER':<12} {'α':>6} {'β':>6} {'MEAN':>7} {'CONFIDENCE':>12} {'OBSERVATIONS':>13}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*7} {'-'*12} {'-'*13}")
    for arm in arms:
        bar_len = int(arm["confidence"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(
            f"  {arm['tier']:<12} {arm['alpha']:>6.1f} {arm['beta']:>6.1f} "
            f"  {arm['mean_reward']:>5.3f}  [{bar}] {arm['confidence']:>4.0%}  "
            f"  {arm['total_obs']:>6.1f} obs"
        )


def print_response(label: str, r, indent=2):
    pad = " " * indent
    print(f"\n{pad}  [{label}]")
    print(f"{pad}  Model:    {r.model_used.split('/')[-1]}  ({r.model_used.split('/')[0]})")
    print(f"{pad}  Task:     {r.task_type}")
    print(f"{pad}  Quality:  {r.quality_score:.3f}")
    print(f"{pad}  Cost:     ${r.cost_usd:.6f}")
    print(f"{pad}  Reward:   {r.goal_match.get('overall_reward', 0):.3f}")
    print(f"{pad}  Answer:   {r.text[:100]}{'...' if len(r.text) > 100 else ''}")


# ─────────────────────────────────────────────────────────────────────────────
banner("BEFORE Phase 3 — LinUCB")
print("""
  LinUCB computes a deterministic score:
    UCB = x·θ + α·√(x·V⁻¹·x)

  Problems:
    - Linear model: assumes reward is linear in context features
    - Needs ~50+ calls before it has enough data to be reliable
    - No natural "confidence" you can show users
    - Same deterministic answer every time for same context
""")

# ─────────────────────────────────────────────────────────────────────────────
banner("AFTER Phase 3 — Thompson Sampling")
print("""
  Thompson Sampling maintains Beta(α, β) per tier:
    - α = weighted successes (high-reward calls)
    - β = weighted failures  (low-reward calls)
    - Each call: SAMPLE from Beta → pick highest → update

  Warm start from metadata (before any org data):
    SLM:      Beta(2, 2)  — neutral, explore early
    Mid:      Beta(3, 2)  — slight positive prior
    Frontier: Beta(5, 1)  — strong prior, expensive but reliable

  Benefits:
    ✓  Smart routing from call #1 (not call #50)
    ✓  Natural confidence score per tier
    ✓  Stochastic = explores automatically, no tuning needed
    ✓  Proven at scale (Netflix, Google, Spotify)
""")

# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 1 — Initial state (warm start, no calls yet)")

try:
    key = create_key("thompson-demo")
except Exception as e:
    print(f"  [ERROR] {e}")
    print("  Start gateway: cd gateway && uvicorn main:app --reload --port 8000")
    sys.exit(1)

agent = X25(
    api_key=key,
    gateway_url=GATEWAY,
    optimize_for={"cost": 0.5, "quality": 0.4, "latency": 0.1},
)

state = get_thompson_state(agent.org)
print(f"\n  Org: {agent.org}")
print(f"  Algorithm: {state['algorithm']}")
print(f"\n  Before any calls — priors from model metadata:")
print_thompson_state(state)

# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 2 — Watch Beta distributions update in real time")

prompts = [
    ("Classify this email as spam or not spam: 'You won a prize!'", "classification"),
    ("Write a Python function to reverse a linked list.",            "code"),
    ("Summarise the key ideas in the theory of evolution.",          "summary"),
    ("What are the ethical implications of autonomous weapons?",     "reasoning"),
    ("Extract all dates from: 'Meeting on Jan 3, deadline Feb 14'", "extraction"),
]

print(f"\n  Sending {len(prompts)} calls. Watch α/β shift after each one.\n")

for i, (prompt, hint) in enumerate(prompts):
    print(f"\n  ── Call {i+1}/{len(prompts)}: {hint} task ──")
    try:
        r = agent.complete(prompt, hint=hint)
        print_response(f"call {i+1}", r)
    except Exception as e:
        print(f"  [ERROR] {e}")
        continue

    state = get_thompson_state(agent.org)
    print(f"\n  Beta state after call {i+1}:")
    print_thompson_state(state)
    time.sleep(0.3)

# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 3 — Two orgs, different task mixes, diverging strategies")

print("\n  Research org: quality-heavy (0.1 cost / 0.8 quality / 0.1 latency)")
print("  DevOps org:   cost-heavy   (0.8 cost / 0.1 quality / 0.1 latency)\n")

key_research = create_key("research-lab")
key_devops   = create_key("devops-team")

agent_research = X25(api_key=key_research, gateway_url=GATEWAY,
                     optimize_for={"cost": 0.1, "quality": 0.8, "latency": 0.1})
agent_devops   = X25(api_key=key_devops,   gateway_url=GATEWAY,
                     optimize_for={"cost": 0.8, "quality": 0.1, "latency": 0.1})

# Both send the same prompt — different routing due to different optimize_for
shared_prompt = "Explain gradient descent in machine learning."

print("  Sending same prompt to both orgs...")
try:
    r_research = agent_research.complete(shared_prompt, hint="reasoning")
    r_devops   = agent_devops.complete(shared_prompt, hint="reasoning")

    print(f"\n  research-lab → {r_research.model_used.split('/')[-1]}  "
          f"(quality={r_research.quality_score:.2f}, cost=${r_research.cost_usd:.6f})")
    print(f"  devops-team  → {r_devops.model_used.split('/')[-1]}  "
          f"(quality={r_devops.quality_score:.2f},  cost=${r_devops.cost_usd:.6f})")

    print("\n  research-lab Beta state:")
    print_thompson_state(get_thompson_state(agent_research.org))
    print("\n  devops-team Beta state:")
    print_thompson_state(get_thompson_state(agent_devops.org))
except Exception as e:
    print(f"  [ERROR] {e}")

# ─────────────────────────────────────────────────────────────────────────────
banner("Phase 3 COMPLETE")
print(f"""
  What changed:
    ✓  LinUCB replaced with Thompson Sampling (gateway/thompson.py)
    ✓  Each tier has a Beta(α, β) distribution — interpretable state
    ✓  Warm-started from model metadata — smart from call #1
    ✓  BaRP partial feedback preserved — only dispatched arm updated
    ✓  New endpoint: GET /thompson/{{org}} — live Bayesian state

  What this means for researchers:
    Every routing decision has a confidence score.
    You can query GET /thompson/{{org}} to see exactly what the
    router has learned about your workload — no black box.

  What's next:
    Phase 4 — Stage System
    X25 tracks your call volume and automatically advances through
    improvement stages, notifying you when fine-tuning is possible.

  Dashboard:  http://localhost:8000/dashboard
  TS state:   http://localhost:8000/thompson/{agent.org}
{DIVIDER}
""")
