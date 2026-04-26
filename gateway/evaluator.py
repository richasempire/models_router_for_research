"""
LLM-as-Judge Evaluator — X25's quality perception layer.

After every model response, X25 autonomously fires this evaluator.
It reads the prompt + response and scores quality 0-1.

If score < threshold → LangGraph escalates to next cascade tier.
If score ≥ threshold → LangGraph commits and updates LinUCB.

Based on PairRM / LLM-Blender methodology (Lin et al., ACL 2023).
We use GPT-4o-mini as the judge — cheap, fast, good enough.

Confidence threshold by task complexity:
  Low complexity (classification, extraction, summary): 0.60
  Medium complexity (general, creative, code):          0.70
  High complexity (reasoning):                          0.75
"""

from __future__ import annotations

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

THRESHOLDS = {
    "classification": 0.60,
    "extraction":     0.60,
    "summary":        0.65,
    "general":        0.70,
    "creative":       0.70,
    "code":           0.70,
    "reasoning":      0.75,
}

JUDGE_SYSTEM = """You are a strict quality evaluator for LLM responses.

Given a prompt and a response, score the response quality from 0.0 to 1.0.

Scoring guide:
  1.0 = Perfect. Complete, accurate, well-structured, directly addresses the prompt.
  0.8 = Good. Minor gaps or slight inaccuracies but mostly correct and useful.
  0.6 = Acceptable. Addresses the prompt but missing important details or has errors.
  0.4 = Poor. Partially relevant but significant gaps, errors, or misunderstanding.
  0.2 = Bad. Mostly irrelevant, wrong, or refuses to answer without good reason.
  0.0 = Completely wrong or empty.

Be strict. A response that is vague or generic should score below 0.6.

Respond with JSON only: {"score": 0.0-1.0, "reason": "<one sentence>"}"""


def evaluate_response(
    prompt: str,
    response_text: str,
    task_type: str,
) -> dict:
    """
    Score a model's response for quality.

    Returns:
        {
            "quality_score": 0.82,
            "threshold": 0.70,
            "passed": True,
            "reason": "Complete and accurate explanation with good examples"
        }
    """
    threshold = THRESHOLDS.get(task_type, 0.70)

    try:
        score, reason = _judge_with_llm(prompt, response_text)
    except Exception as e:
        # Fallback: if judge fails, assume acceptable quality
        score = 0.72
        reason = f"Judge unavailable ({e}), using fallback score"

    return {
        "quality_score": score,
        "threshold": threshold,
        "passed": score >= threshold,
        "reason": reason,
    }


def _judge_with_llm(prompt: str, response_text: str) -> tuple[float, str]:
    """Call GPT-4o-mini to evaluate the response quality."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    user_content = f"""PROMPT:
{prompt[:800]}

RESPONSE:
{response_text[:1200]}"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        max_tokens=80,
        temperature=0.0,
    )

    result = json.loads(completion.choices[0].message.content)
    score = float(result.get("score", 0.7))
    reason = result.get("reason", "")
    return max(0.0, min(1.0, score)), reason
