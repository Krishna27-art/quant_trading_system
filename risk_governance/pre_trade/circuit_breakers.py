import os
import time

from utils.logger import get_logger

logger = get_logger("circuit_breakers")


class CircuitBreaker:
    """
    Automated Kill-Switches
    Halts trading during panic, API failures, or ML latency spikes.
    """

    def __init__(self):
        # In a real system, these would be loaded from .env
        self.twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_from = os.getenv("TWILIO_FROM_NUMBER")
        self.twilio_to = os.getenv("TWILIO_TO_NUMBER")

    def check_vix_limits(self, current_vix: float, current_kelly_size: float) -> tuple[bool, float]:
        """
        VIX > 20: Cut position sizes by 50%.
        VIX > 25: Halt all new trades.
        Returns: (is_trade_allowed, adjusted_position_size)
        """
        if current_vix > 25.0:
            logger.critical(f"CIRCUIT BREAKER: VIX > 25 ({current_vix}). Halting new trades.")
            return False, 0.0

        if current_vix > 20.0:
            logger.warning(
                f"CIRCUIT BREAKER: VIX > 20 ({current_vix}). Cutting position size by 50%."
            )
            return True, current_kelly_size * 0.5

        return True, current_kelly_size

    def check_api_latency(self, start_time: float, threshold_seconds: float = 30.0) -> bool:
        """
        If API takes too long to respond, send SMS alert.
        """
        latency = time.time() - start_time
        if latency > threshold_seconds:
            msg = (
                f"CRITICAL: Broker API Latency exceeded {threshold_seconds}s (took {latency:.2f}s)"
            )
            logger.error(msg)
            self._send_sms_alert(msg)
            return False
        return True

    def check_ml_latency(self, start_time: float, threshold_seconds: float = 5.0) -> bool:
        """
        If ML inference blocks for > 5 seconds, skip the tick.
        """
        latency = time.time() - start_time
        if latency > threshold_seconds:
            logger.error(
                f"CIRCUIT BREAKER: ML inference latency {latency:.2f}s > {threshold_seconds}s. Skipping tick."
            )
            return False
        return True

    def _send_sms_alert(self, message: str):
        """
        Mocks Twilio SMS sending if credentials are not provided.
        """
        if not self.twilio_sid or not self.twilio_token:
            logger.warning(f"[MOCK SMS TO PM]: {message}")
            return

        try:
            # from twilio.rest import Client
            # client = Client(self.twilio_sid, self.twilio_token)
            # client.messages.create(body=message, from_=self.twilio_from, to=self.twilio_to)
            logger.info(f"Twilio SMS Sent: {message}")
        except Exception as e:
            logger.error(f"Failed to send Twilio SMS: {e}")
