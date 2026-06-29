import atexit
import logging
import logging.handlers
import queue
import sys

import structlog

_listener = None


def get_structured_logger(name: str):
    """
    Returns a structlog JSON logger optimized for Grafana/CloudWatch.
    Replaces standard logging and print statements.
    Uses an asynchronous QueueHandler to prevent stdout formatting/IO from blocking.
    """
    global _listener

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    root_logger = logging.getLogger()

    # Configure QueueHandler/QueueListener if not already done
    if not root_logger.handlers:
        log_queue = queue.Queue(-1)
        queue_handler = logging.handlers.QueueHandler(log_queue)
        root_logger.addHandler(queue_handler)
        root_logger.setLevel(logging.INFO)

        # Target handler (writes to stdout)
        console_handler = logging.StreamHandler(sys.stdout)

        # Start listener to process logs asynchronously in a background thread
        _listener = logging.handlers.QueueListener(
            log_queue, console_handler, respect_handler_level=True
        )
        _listener.start()

        # Stop listener on exit
        atexit.register(_listener.stop)

    return structlog.get_logger(name)
