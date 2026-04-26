"""
Research Agent — live hackathon demo.

This is what a developer writes when they use X25.
Notice: zero model names. Zero provider logic. Zero routing code.
X25 handles all of that autonomously.

Run this while the dashboard is open at http://localhost:8000/dashboard
and watch the agent loop fire in real time.

Usage:
    cd demo
    python research_agent.py
    python research_agent.py "quantum computing"
    python research_agent.py "climate change" --org acme-corp --cost 0.8
"""

from __future__ import annotations

import sys
import os
import time
import argparse

# Add x25 SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from x25 import X25, X25Response


def print_response(label: str, r: X25Response, indent: int = 2):
    pad = " " * indent
    print(f"\n{pad}📦 {label}")
    print(f"{pad}   Model:    {r.model_used.split('/')[-1]}  [{r.tier if hasattr(r, 'tier') else ''}]")
    print(f"{pad}   Task:     {r.task_type}")
    print(f"{pad}   Quality:  {r.quality_score:.3f}")
    print(f"{pad}   Cost:     ${r.cost_usd:.6f}  (cascade steps: {r.cascade_steps})")
    print(f"{pad}   Latency:  {r.latency_ms:.0f}ms")
    print(f"{pad}   Audit:    {r.audit_hash[:20]}...")
    print(f"{pad}   Goal:     cost={r.goal_match.get('cost', 0):.2f}  "
          f"quality={r.goal_match.get('quality', 0):.2f}  "
          f"latency={r.goal_match.get('latency', 0):.2f}")
    print(f"\n{pad}   {r.text[:300]}{'...' if len(r.text) > 300 else ''}")


def run_research_agent(
    topic: str,
    org: str = "demo-org",
    optimize_for: dict | None = None,
):
    """
    A 3-step research agent powered by X25.

    Step 1: Decompose the topic into subtopics  (classification task → SLM likely)
    Step 2: Research each subtopic              (reasoning task → may escalate)
    Step 3: Synthesize into a final report      (summary task → mid-tier likely)
    """
    optimize_for = optimize_for or {"cost": 0.5, "quality": 0.4, "latency": 0.1}

    print(f"\n{'='*60}")
    print(f"  X25 Research Agent")
    print(f"  Topic:    {topic}")
    print(f"  Org:      {org}")
    print(f"  Optimize: cost={optimize_for['cost']}  "
          f"quality={optimize_for['quality']}  "
          f"latency={optimize_for.get('latency', 0.1)}")
    print(f"{'='*60}")
    print("  (Watch the dashboard at http://localhost:8000/dashboard)")
    print(f"{'='*60}\n")

    # One agent instance — X25 learns across all calls in this session
    agent = X25(org=org, optimize_for=optimize_for)

    session_cost = 0.0
    session_frontier_cost = 0.0

    # ── Step 1: Decompose ──────────────────────────────────────────────────
    print("STEP 1 — Decompose topic into subtopics")
    print("  (developer wrote: agent.complete(prompt, hint='classification'))")
    print("  X25 autonomously selects the cheapest capable model...\n")

    decompose_r = agent.complete(
        f"Break the topic '{topic}' into exactly 3 specific research subtopics. "
        f"Return them as a numbered list, one per line. Be concise.",
        hint="classification",
    )
    print_response("Decomposition", decompose_r)
    session_cost += decompose_r.cost_usd

    # Parse subtopics from response
    lines = [l.strip() for l in decompose_r.text.strip().split('\n') if l.strip()]
    subtopics = [l.lstrip('0123456789.-) ') for l in lines[:3]]
    if not subtopics:
        subtopics = [f"{topic} fundamentals", f"{topic} applications", f"{topic} future trends"]

    # ── Step 2: Research each subtopic ────────────────────────────────────
    print(f"\n\nSTEP 2 — Research {len(subtopics)} subtopics")
    print("  (developer wrote: agent.complete(prompt) for each subtopic)")
    print("  X25 autonomously routes — harder subtopics may escalate cascade...\n")

    findings = []
    for i, subtopic in enumerate(subtopics):
        print(f"  Researching subtopic {i+1}/{len(subtopics)}: {subtopic}")
        r = agent.complete(
            f"Provide a concise but substantive research summary on: {subtopic}. "
            f"Include key facts, recent developments, and implications. 3-4 sentences.",
        )
        print_response(f"Subtopic {i+1}", r, indent=4)
        findings.append(r.text)
        session_cost += r.cost_usd
        time.sleep(0.5)  # brief pause for dashboard visibility

    # ── Step 3: Synthesize ────────────────────────────────────────────────
    print(f"\n\nSTEP 3 — Synthesize findings into final report")
    print("  (developer wrote: agent.complete(prompt, hint='summary'))")
    print("  X25 autonomously selects a quality model for synthesis...\n")

    combined = "\n\n".join(
        f"Subtopic {i+1} — {subtopics[i]}:\n{f}"
        for i, f in enumerate(findings)
    )

    synth_r = agent.complete(
        f"Synthesize these research findings on '{topic}' into a coherent "
        f"2-paragraph executive summary:\n\n{combined}",
        hint="summary",
    )
    print_response("Final Synthesis", synth_r)
    session_cost += synth_r.cost_usd

    # ── Session summary ───────────────────────────────────────────────────
    # Rough frontier estimate: all calls at Claude Sonnet pricing
    # ($3/M input + $15/M output, assume 300 input + 400 output per call)
    frontier_estimate = 5 * ((300 / 1_000_000 * 3.0) + (400 / 1_000_000 * 15.0))
    saved = max(0, frontier_estimate - session_cost)

    print(f"\n\n{'='*60}")
    print(f"  SESSION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total API calls:      5")
    print(f"  Total cost (X25):     ${session_cost:.6f}")
    print(f"  Est. cost (frontier): ${frontier_estimate:.6f}")
    print(f"  💰 Saved:             ${saved:.6f}  ({saved/max(frontier_estimate,1e-9)*100:.1f}%)")
    print(f"  Dashboard:            http://localhost:8000/dashboard")
    print(f"  Audit trail:          http://localhost:8000/audit/all")
    print(f"  Chain verification:   http://localhost:8000/verify")
    print(f"{'='*60}\n")

    return synth_r.text


def main():
    parser = argparse.ArgumentParser(description="X25 Research Agent Demo")
    parser.add_argument("topic", nargs="?", default="artificial intelligence in healthcare",
                        help="Topic to research")
    parser.add_argument("--org", default="demo-org", help="Organization ID")
    parser.add_argument("--cost", type=float, default=0.5, help="Cost weight (0-1)")
    parser.add_argument("--quality", type=float, default=0.4, help="Quality weight (0-1)")
    parser.add_argument("--latency", type=float, default=0.1, help="Latency weight (0-1)")
    args = parser.parse_args()

    total = args.cost + args.quality + args.latency
    optimize_for = {
        "cost":    args.cost / total,
        "quality": args.quality / total,
        "latency": args.latency / total,
    }

    run_research_agent(topic=args.topic, org=args.org, optimize_for=optimize_for)


if __name__ == "__main__":
    main()
