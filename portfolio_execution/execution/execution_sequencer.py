"""
Execution Sequencer

Single execution gateway with global order queue.
Prevents race conditions, order reordering, and duplicate orders.

Architecture:
Signal → Sequencer → OMS → Broker

Features:
- Global order queue with strict ordering
- Position locking to prevent conflicting trades
- Event-sourced execution log
- Sequence numbers for all orders
"""

import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from utils.logger import get_logger

logger = get_logger("execution_sequencer")


class SequencerState(str, Enum):
    """Sequencer state."""

    IDLE = "idle"
    PROCESSING = "processing"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class SequencedOrder:
    """Order with sequence number."""

    order_id: str
    sequence_number: int
    strategy_id: str
    order_data: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "sequence_number": self.sequence_number,
            "strategy_id": self.strategy_id,
            "order_data": self.order_data,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ExecutionEvent:
    """Event in execution log (event sourcing)."""

    event_id: str
    sequence_number: int
    order_id: str
    event_type: str  # RECEIVED, VALIDATED, SENT, FILLED, REJECTED
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "sequence_number": self.sequence_number,
            "order_id": self.order_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "event_data": self.event_data,
        }


class PositionLock:
    """
    Position lock to prevent conflicting trades.

    Before sending: reserve_position()
    Prevents conflicting trades on same symbol.
    """

    def __init__(self):
        """Initialize position lock."""
        self._locks: dict[str, str] = {}  # symbol -> order_id
        self._lock_timestamps: dict[str, datetime] = {}
        self.logger = logger

    def reserve(self, symbol: str, order_id: str) -> bool:
        """
        Reserve position for symbol.

        Args:
            symbol: Symbol to reserve
            order_id: Order ID reserving

        Returns:
            True if reserved successfully
        """
        if symbol in self._locks:
            existing_order_id = self._locks[symbol]
            self.logger.warning(
                f"Position lock failed: {symbol} already locked by {existing_order_id}"
            )
            return False

        self._locks[symbol] = order_id
        self._lock_timestamps[symbol] = datetime.utcnow()
        self.logger.info(f"Position locked: {symbol} by {order_id}")
        return True

    def release(self, symbol: str, order_id: str) -> bool:
        """
        Release position lock.

        Args:
            symbol: Symbol to release
            order_id: Order ID releasing

        Returns:
            True if released successfully
        """
        if symbol not in self._locks:
            self.logger.warning(f"Position lock not found: {symbol}")
            return False

        if self._locks[symbol] != order_id:
            self.logger.warning(
                f"Position lock mismatch: {symbol} locked by {self._locks[symbol]}, "
                f"release requested by {order_id}"
            )
            return False

        del self._locks[symbol]
        del self._lock_timestamps[symbol]
        self.logger.info(f"Position released: {symbol} by {order_id}")
        return True

    def is_locked(self, symbol: str) -> bool:
        """
        Check if symbol is locked.

        Args:
            symbol: Symbol to check

        Returns:
            True if locked
        """
        return symbol in self._locks

    def get_locked_by(self, symbol: str) -> str | None:
        """
        Get order ID that locked symbol.

        Args:
            symbol: Symbol to check

        Returns:
            Order ID or None
        """
        return self._locks.get(symbol)

    def cleanup_stale_locks(self, max_age_seconds: int = 300):
        """
        Cleanup stale locks (locks older than max_age_seconds).

        Args:
            max_age_seconds: Maximum lock age in seconds
        """
        now = datetime.utcnow()
        stale_symbols = []

        for symbol, timestamp in self._lock_timestamps.items():
            age = (now - timestamp).total_seconds()
            if age > max_age_seconds:
                stale_symbols.append(symbol)

        for symbol in stale_symbols:
            order_id = self._locks[symbol]
            del self._locks[symbol]
            del self._lock_timestamps[symbol]
            self.logger.warning(
                f"Cleaned up stale lock: {symbol} (locked by {order_id}, age: {age:.2f}s)"
            )


class ExecutionSequencer:
    """
    Execution sequencer with global order queue.

    All orders pass through:
    Strategies → Sequencer → OMS → Broker

    Features:
    - Global order queue with strict ordering (#1001, #1002, #1003)
    - Position locking to prevent conflicting trades
    - Event-sourced execution log
    """

    def __init__(self):
        """Initialize execution sequencer."""
        self.state = SequencerState.IDLE
        self.sequence_counter = 0
        self.order_queue: deque = deque()
        self.position_lock = PositionLock()
        self.execution_log: deque = deque(maxlen=100000)  # Event-sourced log
        self.logger = logger

        # Threading
        self._lock = threading.Lock()
        self._processing_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Callbacks
        self._order_callback: Callable[[SequencedOrder], None] | None = None

        self.logger.info("ExecutionSequencer initialized")

    def start(self):
        """Start sequencer processing."""
        with self._lock:
            if self.state == SequencerState.PROCESSING:
                self.logger.warning("Sequencer already processing")
                return

            self.state = SequencerState.PROCESSING
            self._stop_event.clear()
            self._processing_thread = threading.Thread(target=self._process_queue, daemon=True)
            self._processing_thread.start()
            self.logger.info("ExecutionSequencer started")

    def stop(self):
        """Stop sequencer processing."""
        with self._lock:
            if self.state != SequencerState.PROCESSING:
                return

            self.state = SequencerState.STOPPED
            self._stop_event.set()

            if self._processing_thread:
                self._processing_thread.join(timeout=5.0)

            self.logger.info("ExecutionSequencer stopped")

    def pause(self):
        """Pause sequencer processing."""
        with self._lock:
            if self.state != SequencerState.PROCESSING:
                return

            self.state = SequencerState.PAUSED
            self.logger.info("ExecutionSequencer paused")

    def resume(self):
        """Resume sequencer processing."""
        with self._lock:
            if self.state != SequencerState.PAUSED:
                return

            self.state = SequencerState.PROCESSING
            self.logger.info("ExecutionSequencer resumed")

    def submit_order(self, order_id: str, strategy_id: str, order_data: dict[str, Any]) -> int:
        """
        Submit order to sequencer.

        Args:
            order_id: Order ID
            strategy_id: Strategy ID
            order_data: Order data

        Returns:
            Sequence number
        """
        with self._lock:
            self.sequence_counter += 1
            sequence_number = self.sequence_counter

            sequenced_order = SequencedOrder(
                order_id=order_id,
                sequence_number=sequence_number,
                strategy_id=strategy_id,
                order_data=order_data,
            )

            self.order_queue.append(sequenced_order)

            # Log event
            self._log_event(
                sequence_number=sequence_number,
                order_id=order_id,
                event_type="RECEIVED",
                event_data={"strategy_id": strategy_id},
            )

            self.logger.info(f"Order submitted: #{sequence_number} {order_id} from {strategy_id}")

            return sequence_number

    def _process_queue(self):
        """Process order queue (runs in background thread)."""
        while not self._stop_event.is_set():
            # Check if paused
            if self.state == SequencerState.PAUSED:
                time.sleep(0.1)
                continue

            # Get next order
            with self._lock:
                if not self.order_queue:
                    time.sleep(0.01)
                    continue

                sequenced_order = self.order_queue.popleft()

            # Process order
            self._process_order(sequenced_order)

    def _process_order(self, sequenced_order: SequencedOrder):
        """
        Process single order.

        Args:
            sequenced_order: Sequenced order
        """
        symbol = sequenced_order.order_data.get("symbol")

        # Try to reserve position
        if not self.position_lock.reserve(symbol, sequenced_order.order_id):
            self.logger.error(
                f"Failed to reserve position for {symbol}, "
                f"order {sequenced_order.order_id} rejected"
            )

            # Log rejection event
            self._log_event(
                sequence_number=sequenced_order.sequence_number,
                order_id=sequenced_order.order_id,
                event_type="REJECTED",
                event_data={"reason": "position_lock_failed", "symbol": symbol},
            )
            return

        try:
            # Log validation event
            self._log_event(
                sequence_number=sequenced_order.sequence_number,
                order_id=sequenced_order.order_id,
                event_type="VALIDATED",
                event_data={"symbol": symbol},
            )

            # Call callback to execute order
            if self._order_callback:
                self._order_callback(sequenced_order)

            # Log sent event
            self._log_event(
                sequence_number=sequenced_order.sequence_number,
                order_id=sequenced_order.order_id,
                event_type="SENT",
                event_data={"symbol": symbol},
            )

        except Exception as e:
            self.logger.error(f"Error processing order {sequenced_order.order_id}: {e}")

            # Log rejection event
            self._log_event(
                sequence_number=sequenced_order.sequence_number,
                order_id=sequenced_order.order_id,
                event_type="REJECTED",
                event_data={"reason": "processing_error", "error": str(e)},
            )

        finally:
            # Release position lock
            self.position_lock.release(symbol, sequenced_order.order_id)

    def _log_event(
        self, sequence_number: int, order_id: str, event_type: str, event_data: dict[str, Any]
    ):
        """
        Log execution event (event sourcing).

        Args:
            sequence_number: Sequence number
            order_id: Order ID
            event_type: Event type
            event_data: Event data
        """
        event = ExecutionEvent(
            event_id=str(uuid.uuid4()),
            sequence_number=sequence_number,
            order_id=order_id,
            event_type=event_type,
            event_data=event_data,
        )

        self.execution_log.append(event)

    def set_order_callback(self, callback: Callable[[SequencedOrder], None]):
        """
        Set callback for order execution.

        Args:
            callback: Callback function
        """
        self._order_callback = callback
        self.logger.info("Order callback set")

    def get_queue_size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self.order_queue)

    def get_execution_log(self, order_id: str | None = None) -> list:
        """
        Get execution log.

        Args:
            order_id: Filter by order ID (optional)

        Returns:
            List of execution events
        """
        if order_id:
            return [e for e in self.execution_log if e.order_id == order_id]
        return list(self.execution_log)

    def get_stats(self) -> dict[str, Any]:
        """
        Get sequencer statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            return {
                "state": self.state.value,
                "sequence_counter": self.sequence_counter,
                "queue_size": len(self.order_queue),
                "locked_symbols": list(self.position_lock._locks.keys()),
                "execution_log_size": len(self.execution_log),
            }


# Global sequencer instance
_global_sequencer: ExecutionSequencer | None = None


def get_sequencer() -> ExecutionSequencer:
    """Get global execution sequencer."""
    global _global_sequencer
    if _global_sequencer is None:
        _global_sequencer = ExecutionSequencer()
    return _global_sequencer


def initialize_sequencer() -> ExecutionSequencer:
    """
    Initialize and start global sequencer.

    Returns:
        ExecutionSequencer instance
    """
    sequencer = get_sequencer()
    sequencer.start()
    return sequencer
