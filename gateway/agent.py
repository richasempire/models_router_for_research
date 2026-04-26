"""
X25 Routing Agent — LangGraph state machine.

This is the agentic core of X25. Every routing decision runs through
this graph autonomously: no human in the loop, no hardcoded model choices.

Graph nodes (each is a step the agent takes):
  perceive   → read the request, extract metadata
  classify   → identify task type using OpenAI Agents SDK
  policy     → filter eligible models by org constraints
  select     → LinUCB picks the best arm (cascade tier)
  dispatch   → call the selected model via OpenRouter
  judge      → LLM-as-judge scores the response quality
  escalate   → if quality too low, move to next cascade tier
  learn      → update LinUCB with observed reward (BaRP partial feedback)
  audit      → write tamper-evident hash-chained record

Conditional edges (where the agent decides what to do next):
  after judge: quality passed? → learn+audit. failed? → escalate or learn+audit
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional, TypedDict

from langgraph.graph import StateGraph, END

from classifier import classify_task
from evaluator import evaluate_response
from openrouter import OpenRouterClient, CASCADE_TIERS
from linucb import LinUCBRouter, encode_context, compute_reward
from audit import AuditStore


# ── State ─────────────────────────────────────────────────────────────────────
# Everything the agent carries between nodes lives here.

class RoutingState(TypedDict):
    # Input
    prompt: str
    org: str
    optimize_for: dict
    policy: dict
    hint: Optional[str]

    # Perception
    task_type: str
    complexity: float
    classifier_reasoning: str

    # Selection
    context_vector: Any          # numpy array (not JSON-serializable, that's ok)
    linucb_scores: list[float]
    selected_arm: int            # 0=SLM, 1=mid, 2=frontier

    # Cascade tracking
    cascade_steps: int
    tried_arms: list[int]

    # Dispatch result
    response_text: str
    model_used: str
    provider: str
    tier: str
    cost_usd: float
    frontier_cost_usd: float
    latency_ms: float
    carbon_g_co2: float
    prompt_tokens: int
    completion_tokens: int

    # Judgment
    quality_score: float
    quality_threshold: float
    quality_passed: bool
    judge_reason: str

    # Learning
    reward: float
    goal_match: dict

    # Audit
    audit_hash: str

    # WebSocket broadcast (set by main.py before calling agent)
    broadcast_fn: Optional[Any]


# ── Shared singletons ──────────────────────────────────────────────────────────
_openrouter = OpenRouterClient()
_audit = AuditStore()
_linucb_cache: dict[str, LinUCBRouter] = {}


def _get_linucb(org: str) -> LinUCBRouter:
    if org not in _linucb_cache:
        _linucb_cache[org] = LinUCBRouter(org)
    return _linucb_cache[org]


def _broadcast(state: RoutingState, event: str, data: dict):
    """Send live update to dashboard via WebSocket."""
    fn = state.get("broadcast_fn")
    if fn:
        try:
            fn({"event": event, "data": data})
        except Exception:
            pass


# ── Nodes ─────────────────────────────────────────────────────────────────────

def node_perceive(state: RoutingState) -> dict:
    """Initialize cascade tracking."""
    return {
        "cascade_steps": 0,
        "tried_arms": [],
        "selected_arm": 0,
    }


def node_classify(state: RoutingState) -> dict:
    """Classify the task type using OpenAI Agents SDK."""
    result = classify_task(state["prompt"], hint=state.get("hint"))
    _broadcast(state, "classify", {
        "task_type": result["task_type"],
        "complexity": result["complexity"],
        "reasoning": result["reasoning"],
    })
    return {
        "task_type": result["task_type"],
        "complexity": result["complexity"],
        "classifier_reasoning": result["reasoning"],
    }


def node_select(state: RoutingState) -> dict:
    """LinUCB selects the best cascade tier for this context."""
    import numpy as np

    linucb = _get_linucb(state["org"])
    x = encode_context(
        task_type=state["task_type"],
        prompt_tokens=len(state["prompt"].split()),
        optimize_for=state["optimize_for"],
    )

    # Skip tiers already tried (cascade escalation)
    tried = set(state.get("tried_arms", []))
    scores = []
    for i, arm in enumerate(linucb.arms):
        if i in tried:
            scores.append(-999.0)  # exclude tried arms
        else:
            scores.append(arm.ucb_score(x))

    selected_arm = int(np.argmax(scores))
    real_scores = [arm.ucb_score(x) for arm in linucb.arms]

    _broadcast(state, "select", {
        "scores": [round(s, 4) for s in real_scores],
        "selected_arm": selected_arm,
        "tiers": [t["label"] for t in CASCADE_TIERS],
        "tried": list(tried),
    })

    return {
        "context_vector": x,
        "linucb_scores": real_scores,
        "selected_arm": selected_arm,
    }


def node_dispatch(state: RoutingState) -> dict:
    """Call the selected model via OpenRouter."""
    arm = state["selected_arm"]
    tried = state.get("tried_arms", [])

    _broadcast(state, "dispatch", {
        "tier": CASCADE_TIERS[arm]["tier"],
        "model": CASCADE_TIERS[arm]["label"],
        "attempt": len(tried) + 1,
    })

    response = _openrouter.call(
        prompt=state["prompt"],
        tier_index=arm,
    )

    # Compute what frontier would have cost (for savings calculation)
    frontier_cost = _openrouter.frontier_cost_estimate(
        response.prompt_tokens,
        response.completion_tokens,
    )

    return {
        "response_text": response.text,
        "model_used": response.model,
        "provider": response.provider,
        "tier": response.tier,
        "cost_usd": response.cost_usd,
        "frontier_cost_usd": frontier_cost,
        "latency_ms": response.latency_ms,
        "carbon_g_co2": response.carbon_g_co2,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "cascade_steps": state.get("cascade_steps", 0) + 1,
        "tried_arms": tried + [arm],
    }


def node_judge(state: RoutingState) -> dict:
    """LLM-as-judge evaluates response quality."""
    result = evaluate_response(
        prompt=state["prompt"],
        response_text=state["response_text"],
        task_type=state["task_type"],
    )

    _broadcast(state, "judge", {
        "quality_score": result["quality_score"],
        "threshold": result["threshold"],
        "passed": result["passed"],
        "reason": result["reason"],
        "model": state["model_used"],
    })

    return {
        "quality_score": result["quality_score"],
        "quality_threshold": result["threshold"],
        "quality_passed": result["passed"],
        "judge_reason": result["reason"],
    }


def node_learn(state: RoutingState) -> dict:
    """Update LinUCB with observed reward (BaRP partial-feedback)."""
    linucb = _get_linucb(state["org"])
    x = state["context_vector"]

    reward = compute_reward(
        quality_score=state["quality_score"],
        cost_usd=state["cost_usd"],
        latency_ms=state["latency_ms"],
        optimize_for=state["optimize_for"],
        frontier_cost=state["frontier_cost_usd"],
    )

    # Compute goal match: how well did each dimension perform?
    cost_saving = max(0.0, (state["frontier_cost_usd"] - state["cost_usd"])
                      / max(state["frontier_cost_usd"], 1e-9))
    latency_score = max(0.0, 1.0 - state["latency_ms"] / 10_000.0)
    goal_match = {
        "quality": round(state["quality_score"], 3),
        "cost": round(cost_saving, 3),
        "latency": round(latency_score, 3),
        "overall_reward": round(reward, 3),
    }

    # Only update the arm we actually used (BaRP: partial feedback only)
    linucb.update(state["selected_arm"], x, reward)

    _broadcast(state, "learn", {
        "reward": round(reward, 4),
        "goal_match": goal_match,
        "arm_updated": state["selected_arm"],
    })

    return {"reward": reward, "goal_match": goal_match}


def node_audit(state: RoutingState) -> dict:
    """Write tamper-evident audit record."""
    record = _audit.write(
        org=state["org"],
        prompt=state["prompt"],
        task_type=state["task_type"],
        optimize_for=state["optimize_for"],
        linucb_scores=[round(s, 4) for s in state["linucb_scores"]],
        selected_tier=state["tier"],
        model_used=state["model_used"],
        cascade_steps=state["cascade_steps"],
        quality_score=state["quality_score"],
        cost_usd=state["cost_usd"],
        frontier_cost_usd=state["frontier_cost_usd"],
        latency_ms=state["latency_ms"],
        carbon_g_co2=state["carbon_g_co2"],
        goal_match=state.get("goal_match", {}),
    )

    _broadcast(state, "audit", {
        "record_id": record.record_id,
        "hash": record.record_hash[:16] + "...",
        "prev_hash": record.prev_hash[:16] + "...",
        "cost_saved": round(record.cost_saved_usd, 6),
        "carbon_g": round(record.carbon_g_co2, 6),
    })

    return {"audit_hash": record.record_hash}


# ── Conditional edges ──────────────────────────────────────────────────────────

def should_escalate(state: RoutingState) -> str:
    """
    Decision point: escalate to next tier or commit?

    Escalates if:
      - Quality score below threshold AND
      - There are still untried cascade tiers available
    """
    tried = state.get("tried_arms", [])
    quality_passed = state.get("quality_passed", True)
    all_tiers = list(range(len(CASCADE_TIERS)))
    remaining = [i for i in all_tiers if i not in tried]

    if not quality_passed and remaining:
        _broadcast(state, "escalate", {
            "reason": f"quality {state['quality_score']:.2f} < threshold {state['quality_threshold']:.2f}",
            "next_tier": CASCADE_TIERS[remaining[0]]["label"],
        })
        return "escalate"
    return "commit"


# ── Build graph ────────────────────────────────────────────────────────────────

def build_routing_graph():
    """Assemble the LangGraph routing agent."""
    g = StateGraph(RoutingState)

    g.add_node("perceive", node_perceive)
    g.add_node("classify", node_classify)
    g.add_node("select", node_select)
    g.add_node("dispatch", node_dispatch)
    g.add_node("judge", node_judge)
    g.add_node("learn", node_learn)
    g.add_node("audit", node_audit)

    g.set_entry_point("perceive")
    g.add_edge("perceive", "classify")
    g.add_edge("classify", "select")
    g.add_edge("select", "dispatch")
    g.add_edge("dispatch", "judge")

    # Conditional: escalate or commit
    g.add_conditional_edges(
        "judge",
        should_escalate,
        {
            "escalate": "select",   # loop back: pick next best arm
            "commit": "learn",
        },
    )

    g.add_edge("learn", "audit")
    g.add_edge("audit", END)

    return g.compile()


# Build once at import time
routing_graph = build_routing_graph()


def run_routing(
    prompt: str,
    org: str,
    optimize_for: dict,
    policy: dict,
    hint: Optional[str] = None,
    broadcast_fn=None,
) -> dict:
    """
    Entry point for the gateway. Runs the full agentic routing loop.
    Returns the complete routing result as a dict.
    """
    initial_state: RoutingState = {
        "prompt": prompt,
        "org": org,
        "optimize_for": optimize_for,
        "policy": policy,
        "hint": hint,
        "broadcast_fn": broadcast_fn,
        # defaults filled in by nodes
        "task_type": "general",
        "complexity": 0.5,
        "classifier_reasoning": "",
        "context_vector": None,
        "linucb_scores": [],
        "selected_arm": 0,
        "cascade_steps": 0,
        "tried_arms": [],
        "response_text": "",
        "model_used": "",
        "provider": "",
        "tier": "",
        "cost_usd": 0.0,
        "frontier_cost_usd": 0.0,
        "latency_ms": 0.0,
        "carbon_g_co2": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "quality_score": 0.0,
        "quality_threshold": 0.7,
        "quality_passed": False,
        "judge_reason": "",
        "reward": 0.0,
        "goal_match": {},
        "audit_hash": "",
    }

    final = routing_graph.invoke(initial_state)
    return final
