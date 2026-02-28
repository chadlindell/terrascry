"""Tests for the WebSocket endpoint and connection manager."""

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from app.main import app
from app.services.connection_manager import ConnectionManager


class TestConnectionManager:
    def test_init_empty(self):
        cm = ConnectionManager()
        assert cm._channels == {}


def test_websocket_ping():
    """Test WebSocket ping/pong via sync test client."""
    client = TestClient(app)
    with client.websocket_connect("/api/ws/test-channel") as ws:
        ws.send_json({"type": "ping"})
        data = ws.receive_json()
        assert data["type"] == "pong"


def test_websocket_disconnect():
    """Test that connection cleanly disconnects."""
    client = TestClient(app)
    with client.websocket_connect("/api/ws/test-channel") as ws:
        ws.send_json({"type": "ping"})
        ws.receive_json()
    # No error means clean disconnect
