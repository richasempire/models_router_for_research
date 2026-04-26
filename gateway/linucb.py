"""
LinUCB Contextual Bandit — the learning engine of X25.

Based on: Li et al., "A Contextual-Bandit Approach to Personalized
News Article Recommendation," WWW 2010.

Design follows BaRP (arXiv:2510.07429): partial-feedback only.
We only update from the model we actually dispatched to — never
assume what unchosen models would have returned. This mirrors
real production conditions where counterfactuals are unavailable.

How it works:
  - Each cascade tier (SLM / mid / frontier) is an "arm"
  - Context x = [task_type_encoding, token_length_normalized,
                 cost_weight, quality_weight, latency_weight]
  - LinUCB scores each arm: UCB = x·θ + α·√(x·V⁻¹·x)
    └─ first term = exploitation (what we learned works)
    └─ second term = exploration (uncertainty bonus, shrinks with data)
  - We pick the highest UCB, dispatch, observe reward, update
"""

from __future__ import annotations

import json
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# Context vector dimension:
# [task_one_hot x7, log_token_length, cost_w, quality_w, latency_w]
CONTEXT_DIM = 11  # 7 task-type one-hot + log_tokens + cost_w + quality_w + latency_w

# Exploration parameter. Higher = more exploration of uncertain arms.
# 0.5 is conservative (exploit more), 2.0 is aggressive (explore more).
ALPHA = 1.0

# Number of arms = number of cascade tiers
N_ARMS = 3  # SLM, mid, frontier  (free/vlm tracked by registry but not in cascade)

TASK_TYPES = ["code", "reasoning", "summary", "creative",
              "classification", "extraction", "general"]


def encode_context(
    task_type: str,
    prompt_tokens: int,
    optimize_for: dict,
) -> np.ndarray:
    """
    Build the context vector x for LinUCB.

    This vector captures everything relevant about the request and
    the organization's preferences. Same request from two orgs with
    different optimize_for weights gets different routing decisions.
    """
    # One-hot encode task type (7 dims)
    task_vec = np.zeros(len(TASK_TYPES))
    idx = TASK_TYPES.index(task_type) if task_type in TASK_TYPES else -1
    if idx >= 0:
        task_vec[idx] = 1.0

    # Log-normalize token length (1 dim) — log scale avoids huge values
    token_feature = np.log1p(prompt_tokens) / 10.0

    # Optimization weights (3 dims) — org's stated preferences
    cost_w = optimize_for.get("cost", 0.33)
    quality_w = optimize_for.get("quality", 0.34)
    latency_w = optimize_for.get("latency", 0.33)

    x = np.concatenate([task_vec, [token_feature, cost_w, quality_w, latency_w]])
    return x.astype(np.float64)


def compute_reward(
    quality_score: float,
    cost_usd: float,
    latency_ms: float,
    optimize_for: dict,
    frontier_cost: float,
) -> float:
    """
    Convert routing outcome into a scalar reward signal for LinUCB.

    Reward is a weighted combination of normalized objectives,
    matching the organization's stated optimize_for weights.
    Higher reward = this routing decision was better for this org.
    """
    # Normalize cost saving: fraction saved vs frontier (0=no saving, 1=free)
    cost_saving = max(0.0, (frontier_cost - cost_usd) / max(frontier_cost, 1e-9))

    # Normalize latency: invert and cap (faster = higher reward)
    latency_score = max(0.0, 1.0 - (latency_ms / 10_000.0))

    reward = (
        optimize_for.get("quality", 0.33) * quality_score
        + optimize_for.get("cost", 0.33) * cost_saving
        + optimize_for.get("latency", 0.34) * latency_score
    )
    return float(np.clip(reward, 0.0, 1.0))


@dataclass
class ArmState:
    """Per-arm LinUCB state: V matrix and b vector."""
    V: np.ndarray = field(default_factory=lambda: np.eye(CONTEXT_DIM))
    b: np.ndarray = field(default_factory=lambda: np.zeros(CONTEXT_DIM))

    @property
    def theta(self) -> np.ndarray:
        """Current best-estimate weight vector."""
        return np.linalg.solve(self.V, self.b)

    def ucb_score(self, x: np.ndarray, alpha: float = ALPHA) -> float:
        """
        Upper Confidence Bound for this arm given context x.
        UCB = x·θ + α·√(x·V⁻¹·x)
        """
        theta = self.theta
        V_inv = np.linalg.inv(self.V)
        exploitation = float(x @ theta)
        exploration = alpha * float(np.sqrt(x @ V_inv @ x))
        return exploitation + exploration

    def update(self, x: np.ndarray, reward: float):
        """Update arm state with observed (context, reward) pair."""
        self.V += np.outer(x, x)
        self.b += reward * x


class LinUCBRouter:
    """
    Per-organization LinUCB router.

    Each org gets its own set of arm states, so routing learns
    independently per org. Shared initialization (identity matrix)
    gives warm start for new orgs — no cold-start collapse.
    """

    def __init__(self, org: str, state_dir: str = "/tmp/x25_linucb"):
        self.org = org
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.arms: list[ArmState] = self._load_or_init()

    def _state_path(self) -> str:
        safe = self.org.replace("/", "_")
        return os.path.join(self.state_dir, f"{safe}.json")

    def _load_or_init(self) -> list[ArmState]:
        path = self._state_path()
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                return [
                    ArmState(
                        V=np.array(arm["V"]),
                        b=np.array(arm["b"]),
                    )
                    for arm in data
                ]
            except Exception:
                pass
        return [ArmState() for _ in range(N_ARMS)]

    def save(self):
        path = self._state_path()
        with open(path, "w") as f:
            json.dump(
                [{"V": arm.V.tolist(), "b": arm.b.tolist()} for arm in self.arms],
                f,
            )

    def select_arm(self, x: np.ndarray) -> tuple[int, list[float]]:
        """
        Score all arms and return the index with highest UCB.
        Also returns all scores for dashboard transparency.
        """
        scores = [arm.ucb_score(x) for arm in self.arms]
        return int(np.argmax(scores)), scores

    def update(self, arm_index: int, x: np.ndarray, reward: float):
        """
        Update only the selected arm (BaRP partial-feedback design).
        We never update arms we didn't select — no counterfactual assumptions.
        """
        self.arms[arm_index].update(x, reward)
        self.save()

    def get_weights_summary(self) -> list[dict]:
        """Return current learned weights per arm for dashboard display."""
        try:
            from model_registry import get_registry
            registry = get_registry()
            tiers = registry.get_tiers()
            labels = [f"{t.tier} ({t.model.id.split('/')[-1]})" for t in tiers]
        except Exception:
            labels = ["free", "slm", "mid", "frontier"]
        labels = labels[:len(self.arms)]  # guard length mismatch
        while len(labels) < len(self.arms):
            labels.append(f"tier-{len(labels)}")
        return [
            {
                "tier": labels[i],
                "theta_norm": float(np.linalg.norm(arm.theta)),
                "exploration_volume": float(np.linalg.det(arm.V)),
            }
            for i, arm in enumerate(self.arms)
        ]
