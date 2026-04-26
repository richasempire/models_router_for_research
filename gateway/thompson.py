"""
Thompson Sampling Router — Phase 3 upgrade from LinUCB.

Why Thompson Sampling instead of LinUCB?
  LinUCB is a linear model — it assumes reward is a linear function of
  context. That works, but it's rigid. Thompson Sampling is Bayesian:
  each arm has a probability distribution over its true reward, and we
  sample from it rather than computing a deterministic UCB score.

  Result: naturally explores uncertain arms, exploits known good ones,
  and gives you a real "confidence" number you can show users.

How it works (simple version):
  Each cascade tier maintains two numbers: α (wins) and β (losses).
  Think of it as a coin: if we've seen this tier produce good results
  α times and bad results β times, we draw a sample from Beta(α, β).
  The tier with the highest sample wins. Over time, good tiers win
  more often because their Beta distribution shifts right.

Warm start with model metadata:
  Instead of starting all arms at (1,1) — total ignorance — we
  initialise each tier using its known cost profile:
    SLM:      α=2, β=2  (neutral — explore it early)
    Mid:      α=3, β=2  (slight positive prior)
    Frontier: α=5, β=1  (strong prior — it's expensive but reliable)

  This means on call #1, the router already has a sensible opinion
  rather than picking randomly.

Per-org isolation:
  Same design as LinUCBRouter — each org gets its own state file.
  Two orgs with different task mixes converge to different routing
  strategies completely independently.

Reference: Russo et al., "A Tutorial on Thompson Sampling" (2018)
           arXiv:1707.02038
"""

from __future__ import annotations

import json
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


STATE_DIR = os.environ.get("X25_THOMPSON_DIR", "/tmp/x25_thompson")

# Warm-start priors per tier — encodes our prior knowledge about
# model reliability before we've seen any org-specific data.
# Higher α/β ratio = stronger belief it's a good tier.
TIER_PRIORS = {
    "slm":      {"alpha": 2.0, "beta": 2.0},   # neutral — try it early
    "mid":      {"alpha": 3.0, "beta": 2.0},   # slight positive prior
    "frontier": {"alpha": 5.0, "beta": 1.0},   # reliable but expensive
    "vlm":      {"alpha": 3.0, "beta": 2.0},   # same as mid
}

DEFAULT_PRIOR = {"alpha": 2.0, "beta": 2.0}

# Reward threshold: above this = "win", below = "loss"
REWARD_WIN_THRESHOLD = 0.6


@dataclass
class ArmBeta:
    """
    Beta distribution state for one arm (tier).
    Beta(α, β): α = pseudo-successes, β = pseudo-failures.
    """
    tier:  str
    alpha: float
    beta:  float

    def sample(self) -> float:
        """Draw one sample — used for arm selection."""
        return float(np.random.beta(self.alpha, self.beta))

    def mean(self) -> float:
        """Expected reward (exploitation estimate)."""
        return self.alpha / (self.alpha + self.beta)

    def uncertainty(self) -> float:
        """Variance of the Beta distribution — how uncertain we are."""
        n = self.alpha + self.beta
        return (self.alpha * self.beta) / (n * n * (n + 1))

    def confidence(self) -> float:
        """
        Confidence score 0→1: how sure we are about this arm's value.
        High α+β = many observations = high confidence.
        Capped at 1.0 after ~50 effective observations.
        """
        return min(1.0, (self.alpha + self.beta) / 50.0)

    def update(self, reward: float):
        """
        Update from an observed reward.
        Reward above threshold = win (α++), below = loss (β++).
        Partial credit: we scale the update by how far above/below threshold.
        """
        if reward >= REWARD_WIN_THRESHOLD:
            self.alpha += reward                      # stronger win = bigger update
        else:
            self.beta  += (1.0 - reward)              # stronger loss = bigger update


class ThompsonRouter:
    """
    Per-org Thompson Sampling router.

    Replaces LinUCBRouter. Same interface so agent.py needs minimal changes.
    State persisted to JSON after every update.
    """

    def __init__(self, org: str, state_dir: str = STATE_DIR):
        self.org = org
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.arms: list[ArmBeta] = self._load_or_init()

    def _state_path(self) -> str:
        safe = self.org.replace("/", "_").replace(":", "_")
        return os.path.join(self.state_dir, f"{safe}.json")

    def _load_or_init(self) -> list[ArmBeta]:
        path = self._state_path()
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                return [
                    ArmBeta(tier=arm["tier"], alpha=arm["alpha"], beta=arm["beta"])
                    for arm in data
                ]
            except Exception:
                pass
        # Fresh org — warm-start from tier priors
        return self._init_from_registry()

    def _init_from_registry(self) -> list[ArmBeta]:
        """Build arms from live registry cascade tiers with metadata priors."""
        try:
            from model_registry import get_registry
            tiers = get_registry().get_cascade_tiers()
            return [
                ArmBeta(
                    tier=t.tier,
                    alpha=TIER_PRIORS.get(t.tier, DEFAULT_PRIOR)["alpha"],
                    beta=TIER_PRIORS.get(t.tier,  DEFAULT_PRIOR)["beta"],
                )
                for t in tiers
            ]
        except Exception:
            # Fallback if registry not ready
            return [
                ArmBeta(tier="slm",      alpha=2.0, beta=2.0),
                ArmBeta(tier="mid",      alpha=3.0, beta=2.0),
                ArmBeta(tier="frontier", alpha=5.0, beta=1.0),
            ]

    def save(self):
        with open(self._state_path(), "w") as f:
            json.dump(
                [{"tier": a.tier, "alpha": a.alpha, "beta": a.beta}
                 for a in self.arms],
                f,
            )

    # ── Core API (matches LinUCBRouter interface) ──────────────────────────────

    def select_arm(self, tried: Optional[set] = None) -> tuple[int, list[float]]:
        """
        Sample from each arm's Beta distribution, return highest.
        Skips already-tried arms (cascade escalation).
        Returns (arm_index, all_samples).
        """
        tried = tried or set()
        samples = []
        for i, arm in enumerate(self.arms):
            if i in tried:
                samples.append(-1.0)
            else:
                samples.append(arm.sample())

        best = int(np.argmax(samples))
        return best, samples

    def update(self, arm_index: int, reward: float):
        """Update only the dispatched arm (BaRP partial-feedback)."""
        if 0 <= arm_index < len(self.arms):
            self.arms[arm_index].update(reward)
            self.save()

    def scores(self) -> list[float]:
        """Current mean reward estimate per arm — for dashboard display."""
        return [arm.mean() for arm in self.arms]

    def get_state_summary(self) -> list[dict]:
        """Full state per arm for dashboard / demo display."""
        return [
            {
                "tier":        arm.tier,
                "alpha":       round(arm.alpha, 2),
                "beta":        round(arm.beta,  2),
                "mean_reward": round(arm.mean(), 3),
                "uncertainty": round(arm.uncertainty(), 4),
                "confidence":  round(arm.confidence(), 2),
                "total_obs":   round(arm.alpha + arm.beta, 1),
            }
            for arm in self.arms
        ]


# ── Module-level cache (one router per org) ────────────────────────────────────

_cache: dict[str, ThompsonRouter] = {}


def get_thompson(org: str) -> ThompsonRouter:
    if org not in _cache:
        _cache[org] = ThompsonRouter(org)
    return _cache[org]
