"""
OpenRouter client — X25's connection to 300+ models via one API.

Phase 2: CASCADE_TIERS is gone. Model selection is fully dynamic —
the ModelRegistry decides which model maps to each tier at runtime.
The routing agent passes a tier name ("slm", "mid", "frontier", "vlm")
and this client resolves the live best model for that tier.
"""

from __future__ import annotations

import os
import time
import httpx
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

MAX_RETRIES  = 3
RETRY_DELAYS = [1, 3, 8]

# Grid carbon intensity — US East data centers (gCO2 per kWh)
GRID_INTENSITY_G_CO2_PER_KWH = 386.0

# Frontier pricing for savings calculation
# Updated dynamically when registry refreshes, but kept as fallback
_FRONTIER_INPUT_PER_M  = 3.0
_FRONTIER_OUTPUT_PER_M = 15.0


@dataclass
class ModelResponse:
    text: str
    model: str
    provider: str
    tier: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    carbon_g_co2: float


class OpenRouterClient:
    """
    Calls any model on OpenRouter by tier name.
    Model resolution: tier → ModelRegistry → live best model ID.
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set in .env")

    def call(
        self,
        prompt: str,
        tier: str = "slm",
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        model_id: Optional[str] = None,   # override — used by Thompson router
    ) -> ModelResponse:
        """
        Call the best model for the given tier (or an explicit model_id).
        Retries on 429. Returns ModelResponse with real cost from OpenRouter.
        """
        from model_registry import get_registry

        registry = get_registry()

        # Resolve model
        if model_id:
            entry = registry.get_model_by_id(model_id)
            resolved_tier = entry.tier if entry else tier
            resolved_id   = model_id
            energy        = entry.energy_j_per_token if entry else 1.5
            provider      = model_id.split("/")[0]
        else:
            snapshot = registry.get_tier(tier)
            if not snapshot:
                raise RuntimeError(f"No models available for tier '{tier}'")
            entry         = snapshot.model
            resolved_tier = tier
            resolved_id   = entry.id
            energy        = entry.energy_j_per_token
            provider      = entry.provider

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        resp = None
        for attempt in range(MAX_RETRIES):
            start = time.time()
            try:
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(
                        f"{self.BASE_URL}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "https://x25.ai",
                            "X-Title": "X25 Routing Agent",
                        },
                        json={
                            "model": resolved_id,
                            "messages": messages,
                            "max_tokens": max_tokens,
                            "temperature": 0.3,
                        },
                    )
                    if resp.status_code == 429:
                        delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                        print(f"[x25] 429 on {resolved_id}, retrying in {delay}s…")
                        time.sleep(delay)
                        last_error = f"429 after {attempt+1} attempts"
                        continue
                    resp.raise_for_status()
                    break
            except httpx.HTTPStatusError as e:
                last_error = str(e)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAYS[attempt])
                    continue
                raise
        else:
            raise RuntimeError(
                f"OpenRouter {resolved_id} failed after {MAX_RETRIES} retries: {last_error}"
            )

        latency_ms = (time.time() - start) * 1000
        data = resp.json()

        text              = data["choices"][0]["message"]["content"]
        usage             = data.get("usage", {})
        prompt_tokens     = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cost_usd          = float(usage.get("cost", 0.0))

        energy_joules = completion_tokens * energy
        carbon_g_co2  = (energy_joules / 3_600_000) * GRID_INTENSITY_G_CO2_PER_KWH

        return ModelResponse(
            text=text,
            model=resolved_id,
            provider=provider,
            tier=resolved_tier,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            carbon_g_co2=carbon_g_co2,
        )

    def frontier_cost_estimate(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Estimate cost at the current frontier tier.
        Uses live registry pricing if available, falls back to hardcoded.
        """
        try:
            from model_registry import get_registry
            snapshot = get_registry().get_tier("frontier")
            if snapshot:
                inp  = snapshot.model.cost_per_1m_input  / 1_000_000
                out  = snapshot.model.cost_per_1m_output / 1_000_000
                return prompt_tokens * inp + completion_tokens * out
        except Exception:
            pass
        return (prompt_tokens / 1_000_000 * _FRONTIER_INPUT_PER_M) + \
               (completion_tokens / 1_000_000 * _FRONTIER_OUTPUT_PER_M)
