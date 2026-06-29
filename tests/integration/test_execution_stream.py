import multiprocessing
import time

import pytest

from portfolio_execution.config import ExecutionMode, TradingConfig
from portfolio_execution.state_persistence import RedisStateStore
from scripts.execution_worker import start_worker_process


@pytest.mark.integration
def test_execution_stream_fill():
    from portfolio_execution.state_persistence import StatePersistenceConfig

    # 1. Setup config in Paper mode
    config = TradingConfig.from_env()
    config.mode = ExecutionMode.PAPER

    # 2. Connect to Redis and clean streams
    state_config = StatePersistenceConfig(use_redis=True, redis_url="redis://localhost:6379/0")
    state_store = RedisStateStore(config=state_config)
    r = state_store.client
    assert r is not None, "Redis must be running for this test"

    r.delete("stream:orders:pending")
    r.delete("stream:orders:filled")

    # 3. Start execution process worker in background
    # Run for 10 seconds
    p = multiprocessing.Process(target=start_worker_process, args=(config, 10))
    p.start()

    time.sleep(1.0)  # Wait for worker to connect

    # 4. Publish a mock order to stream:orders:pending
    order_id = "test_order_123"
    order_data = {
        "order_id": order_id,
        "symbol": "RELIANCE",
        "side": "BUY",
        "quantity": "10",
        "price": "2500.0",
        "order_type": "LIMIT",
        "stop_price": "0.0",
        "broker_name": "paper",
    }
    r.xadd("stream:orders:pending", order_data)

    # 5. Poll stream:orders:filled for the fill event
    fill_received = False
    start_time = time.time()
    while time.time() - start_time < 5.0:
        streams = r.xread({"stream:orders:filled": "0-0"}, block=1000, count=1)
        if streams:
            for _stream_name, messages in streams:
                for _msg_id, payload in messages:
                    # Decode payload
                    decoded_payload = {}
                    for k, v in payload.items():
                        key = k.decode("utf-8") if isinstance(k, bytes) else k
                        val = v.decode("utf-8") if isinstance(v, bytes) else v
                        decoded_payload[key] = val

                    if decoded_payload.get("order_id") == order_id:
                        assert decoded_payload.get("status") == "FILLED"
                        assert int(decoded_payload.get("filled_qty")) == 10
                        assert float(decoded_payload.get("fill_price")) > 0.0
                        fill_received = True
                        break
            if fill_received:
                break
        time.sleep(0.1)

    # 6. Stop worker
    p.terminate()
    p.join()

    assert fill_received, "Fill was not received in stream:orders:filled within 5 seconds"
