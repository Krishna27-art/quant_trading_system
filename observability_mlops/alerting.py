"""
Alerting & Incident Management

Handles real-time critical alerts via PagerDuty, Slack, or Email.
Used for margin breaches, feed disconnects, and fatal crashes.
"""

import os
from dataclasses import dataclass
from datetime import datetime

import requests

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AlertConfig:
    slack_webhook_url: str = os.getenv("SLACK_WEBHOOK_URL", "")
    pagerduty_routing_key: str = os.getenv("PAGERDUTY_ROUTING_KEY", "")
    alert_level: str = "WARNING"  # Only alert on >= this level


class AlertManager:
    """
    Manages routing of critical alerts to external channels asynchronously.
    """

    def __init__(self, config: AlertConfig = AlertConfig()):
        self.config = config
        import queue
        import threading

        self._alert_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _should_alert(self, level: str) -> bool:
        levels = {"INFO": 0, "WARNING": 1, "CRITICAL": 2, "FATAL": 3}
        return levels.get(level, 0) >= levels.get(self.config.alert_level, 1)

    def _worker(self):
        import queue

        while not self._stop_event.is_set() or not self._alert_queue.empty():
            try:
                alert = self._alert_queue.get(timeout=0.5)
                title, message, level = alert
                if self.config.slack_webhook_url:
                    self._send_slack_alert(title, message, level)
                if level in ["CRITICAL", "FATAL"] and self.config.pagerduty_routing_key:
                    self._send_pagerduty_alert(title, message, level)
                self._alert_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in async alert worker: {e}")

    def send_alert(self, title: str, message: str, level: str = "CRITICAL"):
        """Enqueue alert to be sent asynchronously."""
        logger_func = getattr(logger, level.lower(), logger.error)
        logger_func(f"ALERT [{level}]: {title} - {message}")

        if not self._should_alert(level):
            return

        self._alert_queue.put((title, message, level))

    def stop(self):
        """Shutdown the alert manager gracefully."""
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join()

    def _send_slack_alert(self, title: str, message: str, level: str):
        """Send message to Slack webhook."""
        colors = {
            "INFO": "#36a64f",
            "WARNING": "#ffcc00",
            "CRITICAL": "#ff0000",
            "FATAL": "#000000",
        }
        color = colors.get(level, "#ff0000")

        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": f"[{level}] {title}",
                    "text": message,
                    "footer": "Quant Trading Engine",
                    "ts": int(datetime.now().timestamp()),
                }
            ]
        }

        try:
            requests.post(self.config.slack_webhook_url, json=payload, timeout=5)
            logger.info("Sent Slack alert successfully.")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    def _send_pagerduty_alert(self, title: str, message: str, level: str):
        """Trigger a PagerDuty incident."""
        payload = {
            "routing_key": self.config.pagerduty_routing_key,
            "event_action": "trigger",
            "payload": {
                "summary": f"{title}",
                "severity": "critical",
                "source": "quant-engine",
                "custom_details": {"message": message, "level": level},
            },
        }

        try:
            requests.post("https://events.pagerduty.com/v2/enqueue", json=payload, timeout=5)
            logger.info("Triggered PagerDuty incident successfully.")
        except Exception as e:
            logger.error(f"Failed to trigger PagerDuty: {e}")
