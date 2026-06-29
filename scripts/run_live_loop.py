import os
import sys
import time

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.generate_live_predictions import run
from utils.logger import get_logger

logger = get_logger("live_daemon")


def start_daemon(interval_seconds=300):
    """
    Runs the live predictions generator in an infinite loop.
    Default interval is 5 minutes (300 seconds).
    """
    logger.info(f"Starting Live Data Daemon (Interval: {interval_seconds}s)")

    while True:
        try:
            logger.info("--- Starting new cycle ---")
            run()
            logger.info(f"Cycle complete. Sleeping for {interval_seconds} seconds.")
        except Exception as e:
            logger.error(f"Error in daemon cycle: {e}")
            logger.info("Retrying in 60 seconds...")
            time.sleep(60)
            continue

        time.sleep(interval_seconds)


if __name__ == "__main__":
    # You can change the interval via command line argument
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    start_daemon(interval)
