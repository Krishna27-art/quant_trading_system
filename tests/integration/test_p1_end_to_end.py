import asyncio
import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from api.main import app


@pytest.fixture
def api_client():
    # Set required environment variables for JWT configuration
    os.environ["JWT_SECRET_KEY"] = "test-secret-key-1234567890-test-secret-key"
    os.environ["ADMIN_USERNAME"] = "admin"
    os.environ["ADMIN_PASSWORD"] = "strong-password-123"
    os.environ["REDIS_URL"] = "redis://localhost:6379/0"
    return TestClient(app)


@pytest.mark.integration
def test_jwt_auth_endpoints(api_client):
    # 1. Access protected endpoint without token -> should get 401
    resp = api_client.get("/api/stocks")
    assert resp.status_code == 401

    # 2. Access public health check -> should get 200
    resp = api_client.get("/api/health")
    assert resp.status_code == 200

    # 3. Access public metrics endpoint -> should get 200
    resp = api_client.get("/metrics")
    assert resp.status_code == 200

    # 4. Login with invalid credentials -> should get 401
    resp = api_client.post(
        "/api/auth/login", data={"username": "admin", "password": "wrong-password"}
    )
    assert resp.status_code == 401

    # 5. Login with valid credentials -> should get 200 and access_token
    resp = api_client.post(
        "/api/auth/login", data={"username": "admin", "password": "strong-password-123"}
    )
    assert resp.status_code == 200
    token_data = resp.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"

    # 6. Access protected endpoint with valid token -> should get 200
    token = token_data["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    resp = api_client.get("/api/stocks", headers=headers)
    assert (
        resp.status_code == 200 or resp.status_code == 503
    )  # 503 is acceptable in live environment if database isn't running


@pytest.mark.integration
def test_kill_switch_wiring():
    # Set config & env
    os.environ["REDIS_URL"] = "redis://localhost:6379/0"
    os.environ["KILL_SWITCH_DRY_RUN"] = "false"

    async def _run():
        import redis.asyncio as redis

        r = redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)

        # Clean control system channel first
        pubsub = r.pubsub()
        await pubsub.subscribe("quant:control:system")

        # Execute the kill switch script in a way that triggers direct Redis publication
        from risk_governance.pre_trade.kill_switch import execute_kill_switch

        # Launch execute_kill_switch (should publish trigger command to Redis)
        event = execute_kill_switch(dry_run=False)
        assert event["event"] == "KILL_EXECUTED"

        # Verify the message is present in the pubsub stream
        message = None
        # Wait up to 2 seconds for message to propagate
        for _ in range(20):
            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
            if msg:
                message = msg
                break
            await asyncio.sleep(0.1)

        assert message is not None, "Kill switch command was not published to Redis PubSub"
        data = json.loads(message["data"])
        assert data["command"] == "trigger_kill_switch"

        await pubsub.unsubscribe("quant:control:system")
        await pubsub.close()

    asyncio.run(_run())
