import asyncio
import os
import sys
import time

import redis
from redis.asyncio import from_url as async_from_url

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import contextlib

from portfolio_execution.ems import ExecutionManagementSystem, RoutingStrategy
from portfolio_execution.oms import ManagedOrder, OrderSide, OrderStatus, OrderType
from utils.logger import get_logger

logger = get_logger("execution_worker")


async def run_worker(config=None):
    """
    Background worker that consumes orders from Redis streams and executes them via EMS.
    Implements ack/dedup using Redis Consumer Groups.
    """
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    client = async_from_url(redis_url, decode_responses=True)

    stream_name = "stream:orders:pending"
    group_name = "exec_group"
    consumer_name = f"worker_{os.getpid()}"

    from portfolio_execution.config import ExecutionMode, TradingConfig

    if config is None:
        config = TradingConfig.from_env()

    # Initialize EMS (do not use streams for submission, since we are the consumer)
    ems = ExecutionManagementSystem(
        routing_strategy=RoutingStrategy.FAILOVER,
        order_timeout_seconds=config.broker.order_timeout_seconds,
        max_retries=config.broker.max_retry_attempts,
    )

    # Setup broker connections
    if config.mode in (ExecutionMode.PAPER, ExecutionMode.BACKTEST):
        from portfolio_execution.execution.paper_broker import PaperBrokerAdapter

        paper_broker = PaperBrokerAdapter(slippage_bps=5.0)
        ems.register_broker("paper", paper_broker, is_primary=True)
    elif config.mode == ExecutionMode.LIVE:
        from portfolio_execution.execution.brokers.zerodha_broker import ZerodhaBrokerAdapter

        zerodha_broker = ZerodhaBrokerAdapter()
        ems.register_broker("zerodha", zerodha_broker, is_primary=True)

    await ems.connect_all()

    # Start borrow release consumer task in the background
    async def run_borrow_release_consumer():
        from database.db_async import SessionLocal
        from risk_governance.pre_trade.borrow_manager import release_borrow_by_symbol

        logger.info("Starting out-of-process borrow release consumer...")
        borrow_stream = "stream:borrow:release"
        try:
            await client.xgroup_create(borrow_stream, "borrow_group", id="0", mkstream=True)
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                logger.error(f"Error creating borrow release group: {e}")

        while True:
            try:
                messages = await client.xreadgroup(
                    "borrow_group",
                    f"borrow_worker_{os.getpid()}",
                    {borrow_stream: ">"},
                    count=10,
                    block=100,
                )
                if not messages:
                    await asyncio.sleep(0.1)
                    continue

                for _stream, msgs in messages:
                    for message_id, payload in msgs:
                        symbol = payload.get("symbol")
                        quantity = int(payload.get("quantity", 0))
                        if symbol and quantity > 0:
                            logger.info(
                                f"Releasing borrow out-of-process for {symbol} qty={quantity}"
                            )
                            try:
                                if SessionLocal:
                                    async with SessionLocal() as db:
                                        await release_borrow_by_symbol(db, symbol, quantity)
                            except Exception as db_err:
                                logger.error(
                                    f"Failed to update database for borrow release of {symbol}: {db_err}"
                                )

                        await client.xack(borrow_stream, "borrow_group", message_id)
            except Exception as e:
                logger.error(f"Error in borrow release consumer loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)

    asyncio.create_task(run_borrow_release_consumer())

    loop = asyncio.get_event_loop()

    def handle_fill(order_id, filled_qty, fill_price):
        fill_data = {
            "order_id": str(order_id),
            "status": "FILLED",
            "filled_qty": str(filled_qty),
            "fill_price": str(fill_price),
            "timestamp": str(time.time()),
        }
        loop.create_task(client.xadd("stream:orders:filled", fill_data))
        logger.info(f"Published fill to stream:orders:filled for order {order_id}")

    ems.set_on_fill(handle_fill)

    # Create consumer group
    try:
        await client.xgroup_create(stream_name, group_name, id="0", mkstream=True)
        logger.info(f"Created consumer group {group_name} on {stream_name}")
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e):
            logger.info(f"Consumer group {group_name} already exists.")
        else:
            raise e

    logger.info(f"Execution worker {consumer_name} listening on {stream_name}...")

    while True:
        try:
            # Read from stream
            streams = {stream_name: ">"}
            messages = await client.xreadgroup(
                group_name, consumer_name, streams, count=10, block=100
            )
            if not messages:
                await asyncio.sleep(0.1)
                continue

            for _stream, msgs in messages:
                for message_id, order_data in msgs:
                    logger.info(f"Received order payload: {order_data}")

                    try:
                        # Reconstruct ManagedOrder
                        side_str = order_data.get("side", "BUY").upper()
                        type_str = order_data.get("order_type", "MARKET").upper()

                        order = ManagedOrder(
                            symbol=order_data["symbol"],
                            side=OrderSide[side_str],
                            order_type=OrderType[type_str],
                            quantity=int(float(order_data["quantity"])),
                            price=float(order_data.get("price") or 0.0),
                        )
                        # Overwrite the generated ID with the one from the payload
                        order.order_id = order_data["order_id"]
                        order.status = OrderStatus.VALIDATED

                        # Use EMS to execute it
                        # EMS will handle the routing and broker retry logic
                        success = await ems.submit_order(order)

                        if success:
                            logger.info(f"Successfully executed {order.order_id}")
                        else:
                            logger.error(f"Failed to execute {order.order_id}")

                    except Exception as e:
                        logger.error(f"Error processing order {message_id}: {e}", exc_info=True)

                    # Acknowledge the message so it isn't processed again
                    await client.xack(stream_name, group_name, message_id)
                    logger.info(f"Acked message {message_id}")

        except Exception as e:
            logger.error(f"Error in execution worker loop: {e}", exc_info=True)
            await asyncio.sleep(1.0)


def start_worker_process(config=None, duration_s: int = 10):
    """Sync entrypoint for multiprocessing to run the worker for a fixed duration."""

    async def run_with_timeout():
        worker_task = asyncio.create_task(run_worker(config=config))
        done, pending = await asyncio.wait([worker_task], timeout=float(duration_s))
        if worker_task in pending:
            worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker_task

    asyncio.run(run_with_timeout())


if __name__ == "__main__":
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        logger.info("Execution worker shutting down.")
