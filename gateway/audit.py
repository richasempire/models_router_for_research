"""
Hash-Chained Audit Store — tamper-evident log of every routing decision.

Based on AuditableLLM (MDPI Electronics, Dec 2025) and
Ojewale et al., "Audit Trails for Accountability in LLMs" (arXiv:2601.20727).

Every record contains:
  - Full score decomposition (why X25 chose this model)
  - Actual cost, latency, quality
  - Carbon footprint
  - SHA-256 hash of THIS record + hash of PREVIOUS record

The chain property: if anyone modifies any past record, its hash changes,
which breaks every subsequent record's prev_hash link. Tampering is
immediately detectable by running verify_chain().
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, asdict
from typing import Optional


DB_PATH = "/tmp/x25_audit.db"


@dataclass
class AuditRecord:
    """One routing decision, fully decomposed and cryptographically linked."""
    record_id: str
    timestamp: float
    org: str
    prompt_hash: str             # SHA-256 of prompt (privacy: no raw prompt stored)
    task_type: str
    optimize_for: dict
    linucb_scores: list          # UCB scores for all 3 tiers
    selected_tier: str           # "slm", "mid", or "frontier"
    model_used: str
    cascade_steps: int           # how many models were tried
    quality_score: float
    cost_usd: float
    frontier_cost_usd: float     # what it would have cost at frontier tier
    cost_saved_usd: float        # frontier_cost - actual_cost
    latency_ms: float
    carbon_g_co2: float
    goal_match: dict             # how well this call matched optimize_for
    prev_hash: str               # hash of the previous record (chain link)
    record_hash: str             # SHA-256 of this entire record


def _hash_record(data: dict) -> str:
    """SHA-256 of the record content (excluding record_hash itself)."""
    content = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()


def _hash_prompt(prompt: str) -> str:
    """Store prompt fingerprint, not raw text, for privacy."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


class AuditStore:
    """Append-only, hash-chained SQLite audit log."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    org TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    record_hash TEXT NOT NULL
                )
            """)
            conn.commit()

    def _get_last_hash(self) -> str:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT record_hash FROM audit_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else "genesis"

    def write(
        self,
        org: str,
        prompt: str,
        task_type: str,
        optimize_for: dict,
        linucb_scores: list,
        selected_tier: str,
        model_used: str,
        cascade_steps: int,
        quality_score: float,
        cost_usd: float,
        frontier_cost_usd: float,
        latency_ms: float,
        carbon_g_co2: float,
        goal_match: dict,
    ) -> AuditRecord:
        """Write a new audit record and return it with its hash."""
        prev_hash = self._get_last_hash()
        record_id = f"{int(time.time() * 1000)}-{org[:8]}"
        timestamp = time.time()

        base = {
            "record_id": record_id,
            "timestamp": timestamp,
            "org": org,
            "prompt_hash": _hash_prompt(prompt),
            "task_type": task_type,
            "optimize_for": optimize_for,
            "linucb_scores": linucb_scores,
            "selected_tier": selected_tier,
            "model_used": model_used,
            "cascade_steps": cascade_steps,
            "quality_score": quality_score,
            "cost_usd": cost_usd,
            "frontier_cost_usd": frontier_cost_usd,
            "cost_saved_usd": max(0.0, frontier_cost_usd - cost_usd),
            "latency_ms": latency_ms,
            "carbon_g_co2": carbon_g_co2,
            "goal_match": goal_match,
            "prev_hash": prev_hash,
        }
        record_hash = _hash_record(base)
        base["record_hash"] = record_hash

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO audit_log (record_id, timestamp, org, data_json, record_hash) "
                "VALUES (?, ?, ?, ?, ?)",
                (record_id, timestamp, org, json.dumps(base), record_hash),
            )
            conn.commit()

        return AuditRecord(**base)

    def get_recent(self, org: Optional[str] = None, limit: int = 20) -> list[dict]:
        """Fetch recent audit records for dashboard display."""
        with sqlite3.connect(self.db_path) as conn:
            if org:
                rows = conn.execute(
                    "SELECT data_json FROM audit_log WHERE org = ? "
                    "ORDER BY id DESC LIMIT ?",
                    (org, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT data_json FROM audit_log ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def get_stats(self, org: Optional[str] = None) -> dict:
        """Aggregate stats for dashboard panels."""
        records = self.get_recent(org=org, limit=1000)
        if not records:
            return {
                "total_calls": 0,
                "total_cost_usd": 0.0,
                "total_saved_usd": 0.0,
                "avg_quality": 0.0,
                "avg_latency_ms": 0.0,
                "total_carbon_g": 0.0,
                "model_distribution": {},
                "tier_distribution": {},
                "avg_cascade_steps": 0.0,
            }

        total_cost = sum(r["cost_usd"] for r in records)
        total_saved = sum(r["cost_saved_usd"] for r in records)
        avg_quality = sum(r["quality_score"] for r in records) / len(records)
        avg_latency = sum(r["latency_ms"] for r in records) / len(records)
        total_carbon = sum(r["carbon_g_co2"] for r in records)
        avg_cascade = sum(r["cascade_steps"] for r in records) / len(records)

        model_dist: dict = {}
        tier_dist: dict = {}
        for r in records:
            model_dist[r["model_used"]] = model_dist.get(r["model_used"], 0) + 1
            tier_dist[r["selected_tier"]] = tier_dist.get(r["selected_tier"], 0) + 1

        return {
            "total_calls": len(records),
            "total_cost_usd": round(total_cost, 6),
            "total_saved_usd": round(total_saved, 6),
            "avg_quality": round(avg_quality, 3),
            "avg_latency_ms": round(avg_latency, 1),
            "total_carbon_g": round(total_carbon, 6),
            "model_distribution": model_dist,
            "tier_distribution": tier_dist,
            "avg_cascade_steps": round(avg_cascade, 2),
        }

    def verify_chain(self) -> tuple[bool, str]:
        """
        Verify the entire hash chain is intact.
        Returns (True, "ok") or (False, "broken at record_id=...").
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT data_json FROM audit_log ORDER BY id ASC"
            ).fetchall()

        prev_hash = "genesis"
        for row in rows:
            record = json.loads(row[0])
            if record["prev_hash"] != prev_hash:
                return False, f"broken at record_id={record['record_id']}"
            check = {k: v for k, v in record.items() if k != "record_hash"}
            expected = _hash_record(check)
            if expected != record["record_hash"]:
                return False, f"tampered at record_id={record['record_id']}"
            prev_hash = record["record_hash"]

        return True, "ok"
