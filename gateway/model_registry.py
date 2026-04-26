"""
X25 Model Registry — dynamic discovery from OpenRouter.

Fetches the full model catalog every TTL_SECONDS, clusters models
into 4 capability tiers by completion cost, and picks the best
representative for each tier based on context window + recency.

No hardcoded model names. When OpenRouter adds a new model, X25
discovers it automatically on the next refresh.

Tiers:
  free     — $0/token  (30 models as of today)
  slm      — < $0.30/1M completion tokens
  mid      — $0.30–$3.00/1M
  frontier — > $3.00/1M
  vlm      — any tier, but accepts image input (vision models)
"""

from __future__ import annotations

import os
import time
import threading
import httpx
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

TTL_SECONDS   = 3600          # refresh catalog every hour
MIN_CONTEXT   = 4096          # ignore toy models with tiny context windows
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Cost-per-1M-completion-tokens thresholds
TIER_BOUNDS = {
    "free":     (0.0,   0.0),
    "slm":      (1e-9,  0.30),
    "mid":      (0.30,  3.00),
    "frontier": (3.00,  float("inf")),
}

# Estimated energy per completion token (Joules) — proxy for model size
TIER_ENERGY = {
    "free":     0.4,
    "slm":      0.8,
    "mid":      1.5,
    "frontier": 3.5,
    "vlm":      2.0,
}

# Providers that skip (special routing / audio / image-gen only)
SKIP_IDS = {"openrouter/auto", "openrouter/pareto-code", "openrouter/bodybuilder"}


@dataclass
class ModelEntry:
    id: str                    # e.g. "anthropic/claude-haiku-4-5"
    name: str                  # human label
    provider: str              # e.g. "anthropic"
    tier: str                  # free / slm / mid / frontier / vlm
    cost_per_1m_input: float
    cost_per_1m_output: float
    context_length: int
    is_vision: bool
    energy_j_per_token: float
    created: int               # unix timestamp — higher = newer


@dataclass
class TierSnapshot:
    tier: str
    model: ModelEntry          # best representative for this tier
    candidates: list[ModelEntry] = field(default_factory=list)
    updated_at: float = 0.0


class ModelRegistry:
    """
    Singleton that keeps a live, tiered view of OpenRouter's catalog.
    Thread-safe — background refresh runs every TTL_SECONDS.
    """

    _instance: Optional["ModelRegistry"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._api_key = os.getenv("OPENROUTER_API_KEY", "")
        self._tiers: dict[str, TierSnapshot] = {}
        self._all_models: list[ModelEntry] = []
        self._last_fetch: float = 0.0
        self._refresh_lock = threading.Lock()
        self._refresh()               # eager load at startup
        self._start_background_refresh()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_tier(self, tier: str) -> Optional[TierSnapshot]:
        """Return the current snapshot for a tier name."""
        return self._tiers.get(tier)

    def get_tiers(self) -> list[TierSnapshot]:
        """Return all primary tiers in escalation order."""
        order = ["free", "slm", "mid", "frontier"]
        return [self._tiers[t] for t in order if t in self._tiers]

    def get_cascade_tiers(self) -> list[TierSnapshot]:
        """Return the 3 active cascade tiers (slm → mid → frontier).
        Free/vlm are tracked but excluded from the routing cascade —
        free models are rate-limited; vlm is a separate routing path."""
        order = ["slm", "mid", "frontier"]
        return [self._tiers[t] for t in order if t in self._tiers]

    def get_model_by_id(self, model_id: str) -> Optional[ModelEntry]:
        """Look up a model by its OpenRouter ID."""
        for m in self._all_models:
            if m.id == model_id:
                return m
        return None

    def all_models(self) -> list[ModelEntry]:
        return list(self._all_models)

    def catalog_summary(self) -> dict:
        """Human-readable summary for dashboard / demo display."""
        tiers = self.get_tiers()
        return {
            "total_models": len(self._all_models),
            "last_updated": self._last_fetch,
            "tiers": [
                {
                    "tier": t.tier,
                    "best_model": t.model.id,
                    "best_model_name": t.model.name,
                    "candidates": len(t.candidates),
                    "cost_per_1m_output": t.model.cost_per_1m_output,
                }
                for t in tiers
            ],
        }

    # ── Internal ───────────────────────────────────────────────────────────────

    def _refresh(self):
        """Fetch the OpenRouter catalog and rebuild tier snapshots."""
        with self._refresh_lock:
            try:
                resp = httpx.get(
                    OPENROUTER_MODELS_URL,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    timeout=15.0,
                )
                resp.raise_for_status()
                raw_models = resp.json().get("data", [])
            except Exception as e:
                print(f"[registry] fetch failed: {e} — keeping previous catalog")
                return

            entries = []
            for m in raw_models:
                entry = self._parse(m)
                if entry:
                    entries.append(entry)

            self._all_models = entries
            self._tiers = self._cluster(entries)
            self._last_fetch = time.time()
            summary = self.catalog_summary()
            print(f"[registry] refreshed — {summary['total_models']} models across {len(summary['tiers'])} tiers")
            for t in summary["tiers"]:
                print(f"  {t['tier']:12s} → {t['best_model']:50s}  ${t['cost_per_1m_output']:.3f}/1M  ({t['candidates']} candidates)")

    def _parse(self, raw: dict) -> Optional[ModelEntry]:
        """Parse one OpenRouter model entry into a ModelEntry."""
        model_id = raw.get("id", "")
        if model_id in SKIP_IDS:
            return None
        if not model_id or "/" not in model_id:
            return None

        # Parse pricing — OpenRouter gives per-token strings
        pricing = raw.get("pricing", {})
        try:
            cost_in  = float(pricing.get("prompt",     0) or 0) * 1_000_000
            cost_out = float(pricing.get("completion", 0) or 0) * 1_000_000
        except (ValueError, TypeError):
            return None

        # Skip models with nonsensical negative pricing
        if cost_in < 0 or cost_out < 0:
            return None

        context = raw.get("context_length", 0) or 0
        if context < MIN_CONTEXT:
            return None

        arch = raw.get("architecture", {})
        input_modalities = arch.get("input_modalities", [])
        is_vision = "image" in input_modalities

        provider = model_id.split("/")[0]
        tier = self._assign_tier(cost_out, is_vision)

        return ModelEntry(
            id=model_id,
            name=raw.get("name", model_id),
            provider=provider,
            tier=tier,
            cost_per_1m_input=cost_in,
            cost_per_1m_output=cost_out,
            context_length=context,
            is_vision=is_vision,
            energy_j_per_token=TIER_ENERGY.get(tier, 1.5),
            created=raw.get("created", 0) or 0,
        )

    def _assign_tier(self, cost_out: float, is_vision: bool) -> str:
        if is_vision:
            return "vlm"
        if cost_out == 0.0:
            return "free"
        lo, hi = TIER_BOUNDS["slm"]
        if lo <= cost_out < hi:
            return "slm"
        lo, hi = TIER_BOUNDS["mid"]
        if lo <= cost_out < hi:
            return "mid"
        return "frontier"

    def _cluster(self, entries: list[ModelEntry]) -> dict[str, TierSnapshot]:
        """Group entries into tiers, pick the best representative per tier."""
        groups: dict[str, list[ModelEntry]] = {}
        for e in entries:
            groups.setdefault(e.tier, []).append(e)

        snapshots = {}
        for tier, members in groups.items():
            best = self._pick_best(members, tier)
            snapshots[tier] = TierSnapshot(
                tier=tier,
                model=best,
                candidates=members,
                updated_at=time.time(),
            )

        # Ensure all primary tiers exist — fall back gracefully
        fallbacks = {
            "free":     "meta-llama/llama-3.2-3b-instruct:free",
            "slm":      "openai/gpt-4o-mini",
            "mid":      "anthropic/claude-haiku-4-5",
            "frontier": "anthropic/claude-sonnet-4-6",
        }
        for tier, fallback_id in fallbacks.items():
            if tier not in snapshots:
                fb = self.get_model_by_id(fallback_id)
                if fb:
                    snapshots[tier] = TierSnapshot(tier=tier, model=fb, candidates=[fb])

        return snapshots

    # Trusted providers get a scoring bonus — better reliability and uptime
    TRUSTED_PROVIDERS = {
        "openai": 1.0, "anthropic": 1.0, "google": 0.95,
        "meta-llama": 0.9, "mistralai": 0.9, "deepseek": 0.85,
        "qwen": 0.8, "cohere": 0.8, "amazon": 0.8,
    }

    def _pick_best(self, members: list[ModelEntry], tier: str) -> ModelEntry:
        """
        Score each model in a tier and return the best one.
        Score = 0.3*context + 0.3*recency + 0.4*provider_trust
        Prefer text-only for non-vlm tiers (simpler, more reliable).
        """
        if not members:
            raise ValueError(f"No members for tier {tier}")

        if tier != "vlm":
            text_only = [m for m in members if not m.is_vision]
            pool = text_only if text_only else members
        else:
            pool = members

        max_ctx = max(m.context_length for m in pool) or 1
        max_ts  = max(m.created for m in pool) or 1
        min_ts  = min(m.created for m in pool) or 0

        def score(m: ModelEntry) -> float:
            ctx_score   = m.context_length / max_ctx
            age_range   = max_ts - min_ts or 1
            rec_score   = (m.created - min_ts) / age_range
            trust_score = self.TRUSTED_PROVIDERS.get(m.provider, 0.5)
            return 0.3 * ctx_score + 0.3 * rec_score + 0.4 * trust_score

        return max(pool, key=score)

    def _start_background_refresh(self):
        """Refresh catalog every TTL_SECONDS in a daemon thread."""
        def loop():
            while True:
                time.sleep(TTL_SECONDS)
                self._refresh()
        t = threading.Thread(target=loop, daemon=True)
        t.start()


# ── Module-level singleton ─────────────────────────────────────────────────────

_registry: Optional[ModelRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ModelRegistry()
    return _registry
