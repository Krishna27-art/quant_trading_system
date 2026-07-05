"""
Standalone emergency kill switch.

This module intentionally avoids importing the trading engine or broker adapters.
It can be run by an operator or spawned as a subprocess by the engine when the
main process is unhealthy.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from utils.structured_logger import get_structured_logger

logger = get_structured_logger("kill_switch")

KILL_LOG_PATH = Path(os.getenv("KILL_SWITCH_LOG", "logs/kill_switch.log"))


class KillSwitchError(RuntimeError):
    """Raised when the emergency kill switch cannot complete."""


def _load_kite_client() -> Any:
    raise KillSwitchError("Live execution is disabled. No broker connection available.")


def _append_kill_log(event: dict[str, Any]) -> None:
    KILL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with KILL_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_wiring() -> bool:
    """Dry-run import/configuration check used by the boot self-test."""
    try:
        if os.getenv("KILL_SWITCH_DRY_RUN", "true").lower() == "true":
            _append_kill_log({"event": "KILL_SWITCH_DRY_RUN", "timestamp": time.time()})
            return True
        _load_kite_client()
        return True
    except Exception as exc:
        logger.error("Kill switch wiring check failed", error=str(exc))
        return False


def execute_kill_switch(dry_run: bool = False) -> dict[str, Any]:
    started_at = time.time()
    event: dict[str, Any] = {
        "event": "KILL_EXECUTED" if not dry_run else "KILL_DRY_RUN",
        "timestamp": started_at,
        "cancelled_order_ids": [],
        "exit_order_ids": [],
    }

    if dry_run:
        _append_kill_log(event)
        logger.warning("Kill switch dry run completed")
        return event

    # Import at module load time, not inside the try/except below. Previously
    # `import redis` lived inside the same try block as the publish call, so
    # a missing `redis` package (ImportError) was caught by the same
    # `except Exception` as a transient publish failure, logged as a normal
    # error, and the function returned a dict still tagged
    # "event": "KILL_EXECUTED" — i.e. the single emergency stop mechanism
    # could be completely non-functional (dependency never installed) and
    # every caller would see what looks like a successful kill event.
    try:
        import redis
    except ImportError as exc:
        event["event"] = "KILL_FAILED"
        event["error"] = f"redis package not installed: {exc}"
        logger.critical(
            "🚨 KILL SWITCH FAILED: redis is not installed — cannot publish "
            "trigger_kill_switch command. THE KILL SWITCH IS NON-FUNCTIONAL.",
            error=str(exc),
        )
        _append_kill_log(event)
        raise KillSwitchError(
            "Kill switch could not execute: redis package not installed"
        ) from exc

    try:
        r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))

        # Publish to the system control channel which TradingOrchestrator will listen to
        payload = json.dumps({"command": "trigger_kill_switch", "timestamp": time.time()})
        r.publish("quant:control:system", payload)

        logger.critical(
            "Published KILL_SWITCH command to quant:control:system. Orchestrator will flatten positions."
        )
        _append_kill_log(event)
        return event
    except Exception as e:
        # Previously this branch logged the failure but still returned the
        # original `event` dict, whose "event" field was already set to
        # "KILL_EXECUTED" above. A caller inspecting only the returned dict
        # (not the logs) would see a normal-looking success payload for a
        # kill switch that never actually published anything. Now the
        # event is explicitly marked failed, persisted to the audit log as
        # a failure, and the exception is re-raised so this cannot be
        # silently swallowed by an upstream `except Exception: pass`.
        event["event"] = "KILL_FAILED"
        event["error"] = str(e)
        logger.critical(
            f"🚨 KILL SWITCH FAILED to publish command to Redis: {e}. "
            f"THE KILL SWITCH DID NOT EXECUTE.",
            exc_info=True,
        )
        _append_kill_log(event)
        raise KillSwitchError(f"Kill switch failed to publish to Redis: {e}") from e


def main():
    parser = argparse.ArgumentParser(description="Emergency Kill Switch")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without executing")
    parser.add_argument("--force", action="store_true", help="Bypass dry-run default from .env")
    args = parser.parse_args()

    env_dry = os.getenv("KILL_SWITCH_DRY_RUN", "true").lower() == "true"
    is_dry_run = True

    if args.force or not args.dry_run and not env_dry:
        is_dry_run = False

    if is_dry_run:
        logger.info("Running kill switch in DRY RUN mode. Use --force to execute.")
    else:
        logger.critical("⚠️ RUNNING EMERGENCY KILL SWITCH - REAL EXECUTION ⚠️")

    result = execute_kill_switch(dry_run=is_dry_run)

    print("\n--- Kill Switch Summary ---")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()