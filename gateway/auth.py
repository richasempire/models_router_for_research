"""
X25 Auth — API key management and per-org isolation.

Every org gets a unique key (sk-x25-...). The key:
  - Identifies the org (no need to pass org= separately)
  - Scopes all routing state, audit logs, and bandit learning to that org
  - Rate-limits calls so one tenant can't starve another

Key format: sk-x25-{32 hex chars}
Storage:    SQLite (same machine as gateway, zero deps)
"""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional

DB_PATH = os.environ.get("X25_AUTH_DB", "/tmp/x25_auth.db")


@dataclass
class OrgKey:
    key: str
    org: str
    created_at: float
    last_used: float
    call_count: int
    rate_limit_per_min: int  # 0 = unlimited


class AuthStore:
    """SQLite-backed API key store."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS org_keys (
                    key             TEXT PRIMARY KEY,
                    key_hash        TEXT NOT NULL,
                    org             TEXT NOT NULL,
                    created_at      REAL NOT NULL,
                    last_used       REAL NOT NULL DEFAULT 0,
                    call_count      INTEGER NOT NULL DEFAULT 0,
                    rate_limit_rpm  INTEGER NOT NULL DEFAULT 0
                )
            """)
            # Per-org call log for rate limiting (sliding window)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS call_log (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    org        TEXT NOT NULL,
                    called_at  REAL NOT NULL
                )
            """)
            conn.commit()

    # ── Key lifecycle ──────────────────────────────────────────────────────────

    def create_key(self, org: str, rate_limit_rpm: int = 0) -> str:
        """
        Generate a new API key for an org.
        Returns the raw key — store it safely, we only keep the hash.
        """
        raw_key = "sk-x25-" + secrets.token_hex(16)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        now = time.time()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO org_keys (key, key_hash, org, created_at, last_used, rate_limit_rpm) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (raw_key, key_hash, org, now, now, rate_limit_rpm),
            )
            conn.commit()

        return raw_key

    def validate(self, raw_key: str) -> Optional[OrgKey]:
        """
        Validate a key and return its OrgKey, or None if invalid.
        Also bumps last_used and call_count.
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT key, org, created_at, last_used, call_count, rate_limit_rpm "
                "FROM org_keys WHERE key = ?",
                (raw_key,),
            ).fetchone()

            if not row:
                return None

            # Update usage stats
            conn.execute(
                "UPDATE org_keys SET last_used = ?, call_count = call_count + 1 WHERE key = ?",
                (time.time(), raw_key),
            )
            conn.commit()

        return OrgKey(
            key=row[0],
            org=row[1],
            created_at=row[2],
            last_used=row[3],
            call_count=row[4] + 1,
            rate_limit_per_min=row[5],
        )

    def list_keys(self, org: Optional[str] = None) -> list[dict]:
        """List all keys (redacted) for an org or all orgs."""
        with sqlite3.connect(self.db_path) as conn:
            if org:
                rows = conn.execute(
                    "SELECT key, org, created_at, last_used, call_count, rate_limit_rpm "
                    "FROM org_keys WHERE org = ? ORDER BY created_at DESC",
                    (org,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT key, org, created_at, last_used, call_count, rate_limit_rpm "
                    "FROM org_keys ORDER BY created_at DESC"
                ).fetchall()

        return [
            {
                "key_preview": row[0][:14] + "..." + row[0][-4:],  # sk-x25-abc...ef12
                "org": row[1],
                "created_at": row[2],
                "last_used": row[3],
                "call_count": row[4],
                "rate_limit_rpm": row[5],
            }
            for row in rows
        ]

    def revoke_key(self, raw_key: str) -> bool:
        """Revoke a key. Returns True if it existed."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "DELETE FROM org_keys WHERE key = ?", (raw_key,)
            )
            conn.commit()
            return result.rowcount > 0

    # ── Rate limiting ──────────────────────────────────────────────────────────

    def check_rate_limit(self, org: str, limit_rpm: int) -> tuple[bool, int]:
        """
        Check if org is within rate limit.
        Returns (allowed: bool, calls_in_last_minute: int).
        Cleans up old log entries as a side effect.
        """
        if limit_rpm == 0:
            return True, 0

        window_start = time.time() - 60.0
        with sqlite3.connect(self.db_path) as conn:
            # Purge stale entries
            conn.execute("DELETE FROM call_log WHERE called_at < ?", (window_start,))
            count = conn.execute(
                "SELECT COUNT(*) FROM call_log WHERE org = ? AND called_at >= ?",
                (org, window_start),
            ).fetchone()[0]

            if count >= limit_rpm:
                conn.commit()
                return False, count

            # Log this call
            conn.execute(
                "INSERT INTO call_log (org, called_at) VALUES (?, ?)",
                (org, time.time()),
            )
            conn.commit()

        return True, count + 1

    # ── Stage tracking (Phase 4 preview) ──────────────────────────────────────

    def get_org_call_count(self, org: str) -> int:
        """Total calls made by this org — used for stage advancement."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(call_count), 0) FROM org_keys WHERE org = ?",
                (org,),
            ).fetchone()
        return row[0] if row else 0


# Module-level singleton
_store: Optional[AuthStore] = None


def get_auth_store() -> AuthStore:
    global _store
    if _store is None:
        _store = AuthStore()
    return _store


def extract_key_from_header(authorization: Optional[str]) -> Optional[str]:
    """Parse 'Bearer sk-x25-...' → 'sk-x25-...'"""
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None
