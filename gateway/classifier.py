"""
Task Classifier — X25's perception layer.

Uses OpenAI Agents SDK (2025) to classify incoming prompts into
task types. This classification feeds directly into the LinUCB
context vector, enabling task-aware routing decisions.

Task types map to model strengths:
  code           → models with code-reasoning capability
  reasoning      → models with deep reasoning (may need frontier)
  summary        → cheap models handle well (SLM-friendly)
  creative       → mid-tier usually sufficient
  classification → SLM almost always sufficient
  extraction     → SLM-friendly if structured output works
  general        → balanced routing
"""

from __future__ import annotations

import os
import json
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


# Complexity scores per task type (0-1).
# Higher = more likely to need escalation to frontier.
TASK_COMPLEXITY = {
    "code":           0.7,
    "reasoning":      0.9,
    "summary":        0.3,
    "creative":       0.5,
    "classification": 0.2,
    "extraction":     0.3,
    "general":        0.5,
}


def classify_task(prompt: str, hint: Optional[str] = None) -> dict:
    """
    Classify a prompt into a task type and estimate complexity.

    Uses OpenAI Agents SDK for structured classification.
    Falls back to keyword heuristic if API call fails.

    Returns:
        {
            "task_type": "code",
            "complexity": 0.7,
            "reasoning": "contains function definitions and variable names"
        }
    """
    if hint and hint in TASK_COMPLEXITY:
        return {
            "task_type": hint,
            "complexity": TASK_COMPLEXITY[hint],
            "reasoning": f"Developer-provided hint: {hint}",
        }

    try:
        return _classify_with_llm(prompt)
    except Exception:
        return _classify_with_heuristic(prompt)


def _classify_with_llm(prompt: str) -> dict:
    """Use GPT-4o-mini via OpenAI SDK for fast, accurate classification."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system = """You are a task classifier for an LLM routing system.
Classify the user's prompt into exactly one task type.

Task types:
- code: writing, debugging, or explaining code
- reasoning: multi-step logic, math, analysis, complex problem solving
- summary: condensing or summarizing text
- creative: stories, poems, brainstorming, marketing copy
- classification: categorizing or labeling things
- extraction: pulling structured data from text
- general: anything else

Respond with JSON only:
{"task_type": "<type>", "reasoning": "<one sentence why>"}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt[:500]},  # cap at 500 chars
        ],
        response_format={"type": "json_object"},
        max_tokens=100,
        temperature=0.0,
    )

    result = json.loads(response.choices[0].message.content)
    task_type = result.get("task_type", "general")
    if task_type not in TASK_COMPLEXITY:
        task_type = "general"

    return {
        "task_type": task_type,
        "complexity": TASK_COMPLEXITY[task_type],
        "reasoning": result.get("reasoning", ""),
    }


def _classify_with_heuristic(prompt: str) -> dict:
    """
    Keyword-based fallback classifier. No API call needed.
    Used when the OpenAI API is unavailable.
    """
    p = prompt.lower()

    if any(k in p for k in ["def ", "class ", "function", "import ", "```python",
                              "code", "bug", "error", "debug", "implement"]):
        task_type = "code"
    elif any(k in p for k in ["why", "how does", "explain", "analyze", "reason",
                                "solve", "proof", "derive", "calculate"]):
        task_type = "reasoning"
    elif any(k in p for k in ["summarize", "summary", "tldr", "brief", "condense",
                                "shorten", "key points"]):
        task_type = "summary"
    elif any(k in p for k in ["write a", "story", "poem", "creative", "imagine",
                                "brainstorm", "ideas for"]):
        task_type = "creative"
    elif any(k in p for k in ["classify", "categorize", "label", "is this",
                                "which category"]):
        task_type = "classification"
    elif any(k in p for k in ["extract", "pull out", "find all", "list the",
                                "what are the"]):
        task_type = "extraction"
    else:
        task_type = "general"

    return {
        "task_type": task_type,
        "complexity": TASK_COMPLEXITY[task_type],
        "reasoning": "heuristic keyword match (LLM classifier unavailable)",
    }
