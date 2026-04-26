"""
X25 Live Demo — for room presentations.

Sends a continuous stream of varied prompts so the dashboard visibly
updates in real time: Thompson arms shift, savings accumulate, stage advances.

Usage:
    python demo/live_demo.py

Keep the dashboard open at http://localhost:8000/dashboard while this runs.
Press Ctrl+C to stop.
"""

from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
from x25 import X25

GATEWAY = "http://localhost:8000"
DELAY   = 2.5   # seconds between calls — slow enough to watch the dashboard

PROMPTS = [
    ("Summarise: neural networks learn by adjusting weights through backpropagation.", "summary"),
    ("Classify: is 'Congratulations! You won a free iPhone!' spam?",                   "classification"),
    ("Extract all company names: Apple, Google, and Microsoft reported strong Q4.",     "extraction"),
    ("What is the capital of France?",                                                  "qa"),
    ("Write a Python function that reverses a string.",                                 "code"),
    ("Summarise: photosynthesis converts sunlight, CO2, and water into glucose.",       "summary"),
    ("Classify: is 'Your account has been compromised — click here' phishing?",         "classification"),
    ("Extract all dates: The meeting is on Jan 5, the deadline is Feb 12.",             "extraction"),
    ("What causes lightning?",                                                          "qa"),
    ("Write a SQL query to find the top 5 customers by total spend.",                   "code"),
    ("Summarise: the French Revolution began in 1789 and ended monarchy rule.",         "summary"),
    ("Classify: is 'Team standup at 10am today' urgent?",                               "classification"),
    ("Extract phone numbers: call 555-1234 or reach us at 800-555-9876.",               "extraction"),
    ("What is the difference between TCP and UDP?",                                     "qa"),
    ("Write a regex to match email addresses.",                                         "code"),
    ("Summarise: black holes form when massive stars collapse under their own gravity.", "summary"),
    ("Classify: is 'Please review this PR when you get a chance' high priority?",       "classification"),
    ("Extract all currencies: the deal is worth $4.2M, €1.8M, and £900K.",             "extraction"),
    ("Explain recursion in one sentence.",                                               "qa"),
    ("Write a function to check if a number is prime.",                                 "code"),
]

TIER_COLORS = {"slm": "\033[34m", "mid": "\033[33m", "frontier": "\033[31m"}
RESET = "\033[0m"
GREEN = "\033[32m"
BOLD  = "\033[1m"
DIM   = "\033[2m"


def create_key(org: str) -> str:
    resp = httpx.post(f"{GATEWAY}/keys/create", json={"org": org}, timeout=10)
    resp.raise_for_status()
    return resp.json()["key"]


def divider():
    print(f"\n{'─' * 60}")


print(f"\n{BOLD}X25 Live Demo{RESET}")
print("Dashboard → http://localhost:8000/dashboard")
print("Watch quality, savings, and Thompson arms update in real time.")
print(f"{DIM}Ctrl+C to stop{RESET}\n")

try:
    key = create_key("live-demo")
except Exception as e:
    print(f"[ERROR] Could not create key: {e}")
    sys.exit(1)

agent = X25(
    api_key=key,
    gateway_url=GATEWAY,
    optimize_for={"cost": 0.5, "quality": 0.4, "latency": 0.1},
)

print(f"Org:  {agent.org}")
print(f"Goal: cost=0.50  quality=0.40  latency=0.10\n")

total_saved = 0.0
call_num    = 0

try:
    while True:
        for prompt, hint in PROMPTS:
            call_num += 1
            print(f"  {DIM}#{call_num:02d}{RESET}  [{hint:<14}]  ", end="", flush=True)

            try:
                r = agent.complete(prompt, hint=hint)
                short  = r.model_used.split("/")[-1]
                tier_c = TIER_COLORS.get(r.selected_tier if hasattr(r, "selected_tier") else "", "")
                saved  = getattr(r, "cost_saved_usd", 0.0)
                total_saved += saved

                print(
                    f"{tier_c}{short:<30}{RESET}"
                    f"  q={r.quality_score:.2f}"
                    f"  {GREEN}saved ${saved:.6f}{RESET}"
                    f"  {DIM}total ${total_saved:.4f}{RESET}"
                )
            except Exception as e:
                print(f"[error] {str(e)[:60]}")

            time.sleep(DELAY)

        divider()
        print(f"  Loop complete — {call_num} calls · ${total_saved:.4f} saved · restarting…")
        divider()
        time.sleep(1)

except KeyboardInterrupt:
    divider()
    print(f"\n  {BOLD}Session complete{RESET}")
    print(f"  Total calls:  {call_num}")
    print(f"  Total saved:  {GREEN}${total_saved:.4f}{RESET} vs always-frontier")
    print(f"\n  Dashboard:    http://localhost:8000/dashboard")
    print(f"  Stage status: http://localhost:8000/stage/{agent.org}")
    print(f"  Thompson:     http://localhost:8000/thompson/{agent.org}\n")
