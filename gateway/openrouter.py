"""
OpenRouter client — X25's connection to 300+ models via one API.

Every routing decision ends here. The model string is the only thing
that changes between cascade tiers. OpenRouter returns real cost in
every response, so we never have to estimate it.
"""

from __future__ import annotations

import os
import time
import httpx
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# The three cascade tiers X25 routes across.
# SLM = small, cheap, fast. Mid = balanced. Frontier = best quality.
CASCADE_TIERS = [
    {
        "tier": "slm",
        "model": "openai/gpt-4o-mini",
        "provider": "openai",
        "label": "GPT-4o Mini (SLM)",
        "energy_j_per_token": 0.8,
    },
    {
        "tier": "mid",
        "model": "anthropic/claude-haiku-4-5",
        "provider": "anthropic",
        "label": "Claude Haiku 4.5 (mid)",
        "energy_j_per_token": 1.5,
    },
    {
        "tier": "frontier",
        "model": "anthropic/claude-sonnet-4-6",
        "provider": "anthropic",
        "label": "Claude Sonnet 4.6 (frontier)",
        "energy_j_per_token": 3.5,
    },
]

MAX_RETRIES = 3
RETRY_DELAYS = [1, 3, 8]  # seconds between retries on 429

# Grid carbon intensity for US East data centers (gCO2 per kWh)
# Using static value — in production, pull from Electricity Maps API
GRID_INTENSITY_G_CO2_PER_KWH = 386.0


@dataclass
class ModelResponse:
    text: str
    model: str
    provider: str
    tier: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float          # real cost from OpenRouter
    latency_ms: float
    carbon_g_co2: float      # grams of CO2 for this call


class OpenRouterClient:
    """Calls any model on OpenRouter. Routing = changing the model string."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set in .env")

    def call(
        self,
        prompt: str,
        tier_index: int = 0,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        """
        Call a model at the given cascade tier index (0=SLM, 1=mid, 2=frontier).
        Retries up to MAX_RETRIES times on 429 rate-limit errors.
        Returns a ModelResponse with the text, real cost, and carbon footprint.
        """
        tier_info = CASCADE_TIERS[tier_index]
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = None
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
                            "model": tier_info["model"],
                            "messages": messages,
                            "max_tokens": max_tokens,
                            "temperature": 0.3,
                        },
                    )
                    if resp.status_code == 429:
                        delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                        print(f"[x25] 429 rate limit on {tier_info['model']}, retrying in {delay}s...")
                        time.sleep(delay)
                        last_error = f"429 after {attempt+1} attempts"
                        continue
                    resp.raise_for_status()
                    break  # success
            except httpx.HTTPStatusError as e:
                last_error = str(e)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAYS[attempt])
                    continue
                raise
        else:
            raise RuntimeError(f"OpenRouter {tier_info['model']} failed after {MAX_RETRIES} retries: {last_error}")

        latency_ms = (time.time() - start) * 1000
        data = resp.json()

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # OpenRouter returns cost in USD — real number, not estimate
        cost_usd = float(usage.get("cost", 0.0))

        # Carbon: tokens × energy_per_token (J) × grid_intensity / 3,600,000
        energy_joules = completion_tokens * tier_info["energy_j_per_token"]
        carbon_g_co2 = (energy_joules / 3_600_000) * GRID_INTENSITY_G_CO2_PER_KWH

        return ModelResponse(
            text=text,
            model=tier_info["model"],
            provider=tier_info["provider"],
            tier=tier_info["tier"],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            carbon_g_co2=carbon_g_co2,
        )

    @staticmethod
    def frontier_cost_estimate(prompt_tokens: int, completion_tokens: int) -> float:
        """
        Estimate what this call would have cost at the frontier tier.
        Used to compute savings vs always-frontier baseline.
        Claude Sonnet 4.6: $3/M input, $15/M output
        """
        return (prompt_tokens / 1_000_000 * 3.0) + (completion_tokens / 1_000_000 * 15.0)
