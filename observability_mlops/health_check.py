"""
Health Check Module

Kubernetes-compatible health checks for the quant platform:
- Database connectivity (PostgreSQL / ClickHouse)
- Broker API reachability
- Data feed freshness
- Disk space
- Memory usage

Returns structured HealthReport for liveness and readiness probes.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from utils.logger import get_logger

logger = get_logger("health_check")


class ComponentStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health result for a single infrastructure component."""

    name: str
    status: ComponentStatus
    latency_ms: float = 0.0
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthReport:
    """Aggregated health report across all components."""

    components: list[ComponentHealth] = field(default_factory=list)
    checked_at_epoch: float = field(default_factory=time.time)

    # ── derived properties ──────────────────────────────────

    @property
    def is_live(self) -> bool:
        """Liveness: True unless every component is unhealthy (process can still recover)."""
        if not self.components:
            return True
        return any(c.status != ComponentStatus.UNHEALTHY for c in self.components)

    @property
    def is_ready(self) -> bool:
        """Readiness: True only when no component is unhealthy."""
        if not self.components:
            return False
        return all(c.status != ComponentStatus.UNHEALTHY for c in self.components)

    @property
    def overall_status(self) -> ComponentStatus:
        if not self.components:
            return ComponentStatus.UNHEALTHY
        statuses = {c.status for c in self.components}
        if ComponentStatus.UNHEALTHY in statuses:
            return ComponentStatus.UNHEALTHY
        if ComponentStatus.DEGRADED in statuses:
            return ComponentStatus.DEGRADED
        return ComponentStatus.HEALTHY

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.overall_status.value,
            "is_live": self.is_live,
            "is_ready": self.is_ready,
            "checked_at": self.checked_at_epoch,
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "latency_ms": round(c.latency_ms, 2),
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.components
            ],
        }


class HealthChecker:
    """Runs health checks against all platform components."""

    def __init__(
        self,
        db_url: str | None = None,
        broker_health_url: str | None = None,
        data_feed_staleness_threshold_s: float = 30.0,
        disk_usage_warning_pct: float = 85.0,
        disk_usage_critical_pct: float = 95.0,
        memory_warning_pct: float = 80.0,
        memory_critical_pct: float = 95.0,
    ) -> None:
        self.db_url = db_url or os.getenv(
            "POSTGRES_URL",
            "postgresql://quant_user:quant_pass@localhost:5432/quant_db",
        )
        self.broker_health_url = broker_health_url or os.getenv("BROKER_HEALTH_URL", "")
        self.data_feed_staleness_threshold_s = data_feed_staleness_threshold_s
        self.disk_usage_warning_pct = disk_usage_warning_pct
        self.disk_usage_critical_pct = disk_usage_critical_pct
        self.memory_warning_pct = memory_warning_pct
        self.memory_critical_pct = memory_critical_pct

        # Externally updated by the data feed layer
        self._last_tick_epoch: float = 0.0

    # ── public API ──────────────────────────────────────────

    def update_last_tick(self, epoch: float) -> None:
        """Called by the feed layer whenever a tick arrives."""
        self._last_tick_epoch = epoch

    def run_all(self) -> HealthReport:
        """Execute every registered check and return a HealthReport."""
        report = HealthReport()
        report.components.append(self._check_database())
        report.components.append(self._check_broker())
        report.components.append(self._check_data_feed())
        report.components.append(self._check_disk_space())
        report.components.append(self._check_memory())
        report.checked_at_epoch = time.time()
        logger.info(
            "Health check complete",
            extra={"status": report.overall_status.value},
        )
        return report

    # ── individual checks ───────────────────────────────────

    def _check_database(self) -> ComponentHealth:
        """Verify database connectivity with a lightweight query."""
        start = time.monotonic()
        try:
            import sqlalchemy

            engine = sqlalchemy.create_engine(
                self.db_url,
                pool_pre_ping=True,
                connect_args={"connect_timeout": 5},
            )
            with engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
            latency = (time.monotonic() - start) * 1000
            engine.dispose()
            return ComponentHealth(
                name="database",
                status=ComponentStatus.HEALTHY,
                latency_ms=latency,
                message="PostgreSQL reachable",
            )
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            logger.error("Database health check failed: %s", exc)
            return ComponentHealth(
                name="database",
                status=ComponentStatus.UNHEALTHY,
                latency_ms=latency,
                message=f"Database unreachable: {exc}",
            )

    def _check_broker(self) -> ComponentHealth:
        """HTTP ping against the broker health endpoint."""
        if not self.broker_health_url:
            return ComponentHealth(
                name="broker",
                status=ComponentStatus.DEGRADED,
                message="Broker health URL not configured",
            )
        start = time.monotonic()
        try:
            import urllib.request

            req = urllib.request.Request(self.broker_health_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                status_code = resp.status
            latency = (time.monotonic() - start) * 1000
            if status_code == 200:
                return ComponentHealth(
                    name="broker",
                    status=ComponentStatus.HEALTHY,
                    latency_ms=latency,
                    message="Broker API reachable",
                )
            return ComponentHealth(
                name="broker",
                status=ComponentStatus.DEGRADED,
                latency_ms=latency,
                message=f"Broker returned HTTP {status_code}",
            )
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            logger.warning("Broker health check failed: %s", exc)
            return ComponentHealth(
                name="broker",
                status=ComponentStatus.UNHEALTHY,
                latency_ms=latency,
                message=f"Broker unreachable: {exc}",
            )

    def _check_data_feed(self) -> ComponentHealth:
        """Check freshness of the last received market tick."""
        if self._last_tick_epoch <= 0:
            return ComponentHealth(
                name="data_feed",
                status=ComponentStatus.DEGRADED,
                message="No ticks received yet",
            )
        lag = time.time() - self._last_tick_epoch
        if lag <= self.data_feed_staleness_threshold_s:
            return ComponentHealth(
                name="data_feed",
                status=ComponentStatus.HEALTHY,
                message=f"Feed lag {lag:.1f}s within threshold",
                details={"lag_seconds": round(lag, 2)},
            )
        return ComponentHealth(
            name="data_feed",
            status=ComponentStatus.UNHEALTHY,
            message=f"Feed lag {lag:.1f}s exceeds {self.data_feed_staleness_threshold_s}s threshold",
            details={"lag_seconds": round(lag, 2)},
        )

    def _check_disk_space(self) -> ComponentHealth:
        """Check disk usage on the data partition."""
        try:
            usage = shutil.disk_usage("/")
            used_pct = (usage.used / usage.total) * 100
            free_gb = usage.free / (1024**3)
            details = {
                "used_pct": round(used_pct, 1),
                "free_gb": round(free_gb, 2),
                "total_gb": round(usage.total / (1024**3), 2),
            }
            if used_pct >= self.disk_usage_critical_pct:
                return ComponentHealth(
                    name="disk",
                    status=ComponentStatus.UNHEALTHY,
                    message=f"Disk {used_pct:.1f}% used — critical",
                    details=details,
                )
            if used_pct >= self.disk_usage_warning_pct:
                return ComponentHealth(
                    name="disk",
                    status=ComponentStatus.DEGRADED,
                    message=f"Disk {used_pct:.1f}% used — warning",
                    details=details,
                )
            return ComponentHealth(
                name="disk",
                status=ComponentStatus.HEALTHY,
                message=f"Disk {used_pct:.1f}% used, {free_gb:.1f} GB free",
                details=details,
            )
        except Exception as exc:
            logger.error("Disk space check failed: %s", exc)
            return ComponentHealth(
                name="disk",
                status=ComponentStatus.DEGRADED,
                message=f"Could not determine disk usage: {exc}",
            )

    def _check_memory(self) -> ComponentHealth:
        """Check system memory usage via /proc/meminfo or psutil fallback."""
        try:
            mem_info = self._read_memory_info()
            used_pct = mem_info["used_pct"]
            details = {
                "used_pct": round(used_pct, 1),
                "available_mb": mem_info.get("available_mb", 0),
                "total_mb": mem_info.get("total_mb", 0),
            }
            if used_pct >= self.memory_critical_pct:
                return ComponentHealth(
                    name="memory",
                    status=ComponentStatus.UNHEALTHY,
                    message=f"Memory {used_pct:.1f}% used — critical",
                    details=details,
                )
            if used_pct >= self.memory_warning_pct:
                return ComponentHealth(
                    name="memory",
                    status=ComponentStatus.DEGRADED,
                    message=f"Memory {used_pct:.1f}% used — warning",
                    details=details,
                )
            return ComponentHealth(
                name="memory",
                status=ComponentStatus.HEALTHY,
                message=f"Memory {used_pct:.1f}% used",
                details=details,
            )
        except Exception as exc:
            logger.error("Memory check failed: %s", exc)
            return ComponentHealth(
                name="memory",
                status=ComponentStatus.DEGRADED,
                message=f"Could not determine memory usage: {exc}",
            )

    @staticmethod
    def _read_memory_info() -> dict[str, float]:
        """Read memory statistics from /proc/meminfo (Linux) or psutil."""
        proc_path = "/proc/meminfo"
        if os.path.exists(proc_path):
            mem: dict[str, int] = {}
            with open(proc_path) as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        mem[key] = int(parts[1])  # kB
            total_kb = mem.get("MemTotal", 1)
            available_kb = mem.get("MemAvailable", mem.get("MemFree", 0))
            used_pct = ((total_kb - available_kb) / total_kb) * 100
            return {
                "used_pct": used_pct,
                "available_mb": available_kb / 1024,
                "total_mb": total_kb / 1024,
            }
        # macOS / fallback via psutil
        try:
            import psutil

            vm = psutil.virtual_memory()
            return {
                "used_pct": vm.percent,
                "available_mb": vm.available / (1024 * 1024),
                "total_mb": vm.total / (1024 * 1024),
            }
        except ImportError:
            # Last resort: use os.sysconf on POSIX
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            total_bytes = pages * page_size
            # Without /proc we cannot know available, report 0% used
            return {
                "used_pct": 0.0,
                "available_mb": total_bytes / (1024 * 1024),
                "total_mb": total_bytes / (1024 * 1024),
            }


# ── FastAPI integration ─────────────────────────────────────────


def create_health_endpoints(checker: HealthChecker):
    """Return (liveness_handler, readiness_handler) async callables for FastAPI."""
    from fastapi.responses import JSONResponse

    async def liveness() -> JSONResponse:
        report = checker.run_all()
        status_code = 200 if report.is_live else 503
        return JSONResponse(content=report.to_dict(), status_code=status_code)

    async def readiness() -> JSONResponse:
        report = checker.run_all()
        status_code = 200 if report.is_ready else 503
        return JSONResponse(content=report.to_dict(), status_code=status_code)

    return liveness, readiness
