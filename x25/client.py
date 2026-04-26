"""
X25 SDK — the only file a developer needs to import.

Usage:
    from x25 import X25

    agent = X25(
        org="my-startup",
        optimize_for={"cost": 0.6, "quality": 0.3, "latency": 0.1},
    )
    response = agent.complete("Explain quantum entanglement simply")
    print(response.text)
    print(response.model_used)   # which model X25 chose
    print(response.cost_usd)     # what it cost
    print(response.audit_hash)   # tamper-evident record hash
"""

from __future__ import annotations

import httpx
from dataclasses import dataclass
from typing import Optional


@dataclass
class X25Response:
    """Everything X25 returns after routing a request."""
    text: str                        # the actual model response
    model_used: str                  # e.g. "gpt-4o-mini"
    provider: str                    # e.g. "openai"
    task_type: str                   # e.g. "code", "reasoning", "summary"
    cost_usd: float                  # actual cost of this call
    latency_ms: float                # how long it took
    quality_score: float             # 0-1, from LLM-as-judge
    cascade_steps: int               # how many models were tried (1=first try)
    audit_hash: str                  # SHA-256 hash of the audit record
    goal_match: dict                 # how well this call matched your optimize_for


class X25:
    """
    Drop-in autonomous LLM router.

    Instead of picking a model yourself, you tell X25 what you're
    optimizing for. X25 routes every call through a cascade of models,
    judges the output quality, escalates if needed, and learns from
    every outcome — getting smarter over time.
    """

    def __init__(
        self,
        org: str = "default",
        optimize_for: Optional[dict] = None,
        policy: Optional[dict] = None,
        gateway_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
    ):
        """
        Args:
            org:          Your organization ID. X25 learns per-org preferences.
            optimize_for: Weights summing to 1.0, e.g.:
                          {"cost": 0.6, "quality": 0.3, "latency": 0.1}
                          Defaults to balanced (0.33 each).
            policy:       Hard constraints, e.g.:
                          {"allowed_providers": ["openai", "anthropic"],
                           "max_cost_per_call_usd": 0.05}
            gateway_url:  Where the X25 gateway is running.
            api_key:      Optional auth key for the gateway.
        """
        self.org = org
        self.optimize_for = optimize_for or {"cost": 0.33, "quality": 0.34, "latency": 0.33}
        self.policy = policy or {}
        self.gateway_url = gateway_url.rstrip("/")
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

        self._normalize_weights()

    def _normalize_weights(self):
        """Ensure weights sum to 1.0."""
        total = sum(self.optimize_for.values())
        if total > 0:
            self.optimize_for = {k: v / total for k, v in self.optimize_for.items()}

    def complete(self, prompt: str, hint: Optional[str] = None) -> X25Response:
        """
        Route a prompt to the best model autonomously.

        Args:
            prompt: The text you want a model to respond to.
            hint:   Optional task type hint to help the classifier.
                    One of: "code", "reasoning", "summary", "creative",
                             "classification", "extraction", "general"
                    If omitted, X25 infers it automatically.

        Returns:
            X25Response with the text, model used, cost, quality, and audit hash.
        """
        payload = {
            "prompt": prompt,
            "org": self.org,
            "optimize_for": self.optimize_for,
            "policy": self.policy,
            "hint": hint,
        }

        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(
                    f"{self.gateway_url}/route",
                    json=payload,
                    headers=self._headers,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError:
            raise RuntimeError(
                "X25 gateway not running. Start it with:\n"
                "  cd gateway && uvicorn main:app --reload"
            )

        return X25Response(
            text=data["text"],
            model_used=data["model_used"],
            provider=data["provider"],
            task_type=data["task_type"],
            cost_usd=data["cost_usd"],
            latency_ms=data["latency_ms"],
            quality_score=data["quality_score"],
            cascade_steps=data["cascade_steps"],
            audit_hash=data["audit_hash"],
            goal_match=data["goal_match"],
        )

    def get_stats(self) -> dict:
        """Return routing stats for this org (for the dashboard)."""
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"{self.gateway_url}/stats/{self.org}",
                headers=self._headers,
            )
            resp.raise_for_status()
            return resp.json()
