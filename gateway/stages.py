"""
X25 Stage System — Phase 4.

Every org progresses through 4 stages as they accumulate calls.
X25 tracks this automatically and tells orgs when they unlock the next level.

Stage 1 — Explore    (0–49 calls)
  X25 is learning your task mix. Thompson Sampling is exploring all tiers.
  No action needed from you.

Stage 2 — Exploit    (50–199 calls)
  Enough data. Thompson Sampling has converged on your task patterns.
  Routing is now personalised to your workload.
  X25 notifies: "Your routing has converged. Here's what we learned."

Stage 3 — Feedback   (200–499 calls)
  X25 has enough call history to identify your domain.
  You can now provide labelled examples or thumbs-up/down to fine-tune
  the routing classifier specifically for your tasks.
  X25 notifies: "Ready for feedback. Provide 50+ examples to improve."

Stage 4 — Fine-tune  (500+ calls)
  X25 takes your logged call history and fine-tunes a lightweight SLM
  (Phase 5) on your exact task patterns. That model re-enters your
  routing pool at near-zero cost.
  X25 notifies: "Fine-tuning unlocked. Run /improve to start."

Weekly drift check:
  A background thread checks each org weekly. If quality dropped >10%
  from their Stage 2 baseline, X25 flags it proactively.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional


DB_PATH = os.environ.get("X25_STAGES_DB", "/tmp/x25_stages.db")

STAGE_THRESHOLDS = {
    1:  0,
    2:  50,
    3:  200,
    4:  500,
}

STAGE_NAMES = {
    1: "Explore",
    2: "Exploit",
    3: "Feedback",
    4: "Fine-tune",
}

STAGE_DESCRIPTIONS = {
    1: "X25 is learning your task mix. Thompson Sampling explores all tiers.",
    2: "Routing has converged on your workload. Personalised routing is active.",
    3: "Enough history to fine-tune the routing classifier. Provide examples to improve.",
    4: "Fine-tuning unlocked. Run POST /improve to start SLM fine-tuning.",
}

STAGE_NEXT_ACTION = {
    1: "Keep making calls. Stage 2 unlocks at 50 calls.",
    2: "Stage 3 unlocks at 200 calls. You can provide feedback examples now.",
    3: "Stage 4 unlocks at 500 calls. Provide labelled examples via POST /feedback.",
    4: "Fine-tuning available. POST /improve/{org} to start.",
}


@dataclass
class OrgStage:
    org: str
    stage: int
    total_calls: int
    stage_entered_at: float
    quality_baseline: float       # avg quality when Stage 2 was entered
    improvement_available: bool
    drift_detected: bool
    last_checked: float


class StageTracker:
    """Tracks stage progression and quality drift per org."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()
        self._start_drift_monitor()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS org_stages (
                    org                  TEXT PRIMARY KEY,
                    stage                INTEGER NOT NULL DEFAULT 1,
                    total_calls          INTEGER NOT NULL DEFAULT 0,
                    stage_entered_at     REAL    NOT NULL,
                    quality_baseline     REAL    NOT NULL DEFAULT 0.0,
                    improvement_available INTEGER NOT NULL DEFAULT 0,
                    drift_detected       INTEGER NOT NULL DEFAULT 0,
                    last_checked         REAL    NOT NULL DEFAULT 0
                )
            """)
            # Store feedback examples submitted by orgs (Stage 3+)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS org_feedback (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    org         TEXT    NOT NULL,
                    prompt      TEXT    NOT NULL,
                    good_model  TEXT    NOT NULL,
                    submitted_at REAL   NOT NULL
                )
            """)
            conn.commit()

    # ── Core API ───────────────────────────────────────────────────────────────

    def get_or_create(self, org: str) -> OrgStage:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT org, stage, total_calls, stage_entered_at, quality_baseline, "
                "improvement_available, drift_detected, last_checked "
                "FROM org_stages WHERE org = ?", (org,)
            ).fetchone()
            if not row:
                now = time.time()
                conn.execute(
                    "INSERT INTO org_stages (org, stage, total_calls, stage_entered_at, "
                    "quality_baseline, improvement_available, drift_detected, last_checked) "
                    "VALUES (?, 1, 0, ?, 0.0, 0, 0, ?)",
                    (org, now, now),
                )
                conn.commit()
                return OrgStage(org=org, stage=1, total_calls=0,
                                stage_entered_at=now, quality_baseline=0.0,
                                improvement_available=False, drift_detected=False,
                                last_checked=now)
        return OrgStage(
            org=row[0], stage=row[1], total_calls=row[2],
            stage_entered_at=row[3], quality_baseline=row[4],
            improvement_available=bool(row[5]), drift_detected=bool(row[6]),
            last_checked=row[7],
        )

    def record_call(self, org: str, quality_score: float) -> OrgStage:
        """
        Called after every routing decision.
        Increments call count, checks for stage advancement.
        """
        state = self.get_or_create(org)
        new_calls = state.total_calls + 1
        new_stage = state.stage
        new_baseline = state.quality_baseline
        improvement_available = state.improvement_available

        # Check for stage advancement
        for stage_num in sorted(STAGE_THRESHOLDS.keys(), reverse=True):
            if new_calls >= STAGE_THRESHOLDS[stage_num] and stage_num > state.stage:
                new_stage = stage_num
                print(f"[stages] org='{org}' advanced to Stage {new_stage} "
                      f"({STAGE_NAMES[new_stage]}) after {new_calls} calls")
                # Set quality baseline at Stage 2 entry
                if new_stage == 2:
                    new_baseline = quality_score
                if new_stage >= 3:
                    improvement_available = True
                break

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE org_stages SET total_calls=?, stage=?, quality_baseline=?, "
                "improvement_available=?, last_checked=? WHERE org=?",
                (new_calls, new_stage, new_baseline,
                 int(improvement_available), time.time(), org),
            )
            conn.commit()

        return OrgStage(
            org=org, stage=new_stage, total_calls=new_calls,
            stage_entered_at=state.stage_entered_at, quality_baseline=new_baseline,
            improvement_available=improvement_available,
            drift_detected=state.drift_detected, last_checked=time.time(),
        )

    def get_status(self, org: str) -> dict:
        """Full status object for dashboard and API."""
        state = self.get_or_create(org)

        # Next threshold
        next_stage = state.stage + 1
        next_threshold = STAGE_THRESHOLDS.get(next_stage)
        calls_to_next = max(0, next_threshold - state.total_calls) if next_threshold else 0

        # Progress within current stage
        current_min = STAGE_THRESHOLDS[state.stage]
        if next_threshold:
            stage_range = next_threshold - current_min
            progress = min(1.0, (state.total_calls - current_min) / max(stage_range, 1))
        else:
            progress = 1.0

        return {
            "org":                   state.org,
            "stage":                 state.stage,
            "stage_name":            STAGE_NAMES[state.stage],
            "stage_description":     STAGE_DESCRIPTIONS[state.stage],
            "next_action":           STAGE_NEXT_ACTION[state.stage],
            "total_calls":           state.total_calls,
            "calls_to_next_stage":   calls_to_next,
            "next_stage":            next_stage if next_threshold else None,
            "next_stage_name":       STAGE_NAMES.get(next_stage),
            "progress_in_stage":     round(progress, 3),
            "quality_baseline":      state.quality_baseline,
            "improvement_available": state.improvement_available,
            "drift_detected":        state.drift_detected,
            "all_stages": [
                {
                    "stage":      s,
                    "name":       STAGE_NAMES[s],
                    "threshold":  STAGE_THRESHOLDS[s],
                    "unlocked":   state.total_calls >= STAGE_THRESHOLDS[s],
                    "active":     state.stage == s,
                }
                for s in [1, 2, 3, 4]
            ],
        }

    def submit_feedback(self, org: str, prompt: str, good_model: str):
        """Store a labelled example from the org (Stage 3+ feature)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO org_feedback (org, prompt, good_model, submitted_at) "
                "VALUES (?, ?, ?, ?)",
                (org, prompt, good_model, time.time()),
            )
            conn.commit()

    def get_feedback_count(self, org: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM org_feedback WHERE org = ?", (org,)
            ).fetchone()
        return row[0] if row else 0

    def list_all_orgs(self) -> list[dict]:
        """List all tracked orgs — for operator dashboard."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT org, stage, total_calls, improvement_available, drift_detected "
                "FROM org_stages ORDER BY total_calls DESC"
            ).fetchall()
        return [
            {
                "org": r[0], "stage": r[1], "total_calls": r[2],
                "improvement_available": bool(r[3]), "drift_detected": bool(r[4]),
            }
            for r in rows
        ]

    # ── Quality drift monitor ─────────────────────────────────────────────────

    def _check_drift(self):
        """
        Weekly check: if avg quality dropped >10% from Stage 2 baseline, flag it.
        Runs in background thread.
        """
        try:
            from audit import AuditStore
            audit = AuditStore()

            with sqlite3.connect(self.db_path) as conn:
                orgs = conn.execute(
                    "SELECT org, stage, quality_baseline FROM org_stages "
                    "WHERE stage >= 2 AND quality_baseline > 0"
                ).fetchall()

            for org, stage, baseline in orgs:
                recent = audit.get_recent(org=org, limit=20)
                if len(recent) < 5:
                    continue
                recent_quality = sum(r["quality_score"] for r in recent) / len(recent)
                drift = baseline - recent_quality
                if drift > 0.10:
                    print(f"[stages] DRIFT DETECTED org='{org}' "
                          f"baseline={baseline:.2f} recent={recent_quality:.2f} "
                          f"drop={drift:.2f}")
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute(
                            "UPDATE org_stages SET drift_detected=1 WHERE org=?", (org,)
                        )
                        conn.commit()
        except Exception as e:
            print(f"[stages] drift check error: {e}")

    def _start_drift_monitor(self):
        def loop():
            while True:
                time.sleep(604800)   # weekly
                self._check_drift()
        t = threading.Thread(target=loop, daemon=True)
        t.start()


# ── Singleton ──────────────────────────────────────────────────────────────────

_tracker: Optional[StageTracker] = None
_tracker_lock = threading.Lock()


def get_stage_tracker() -> StageTracker:
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = StageTracker()
    return _tracker
