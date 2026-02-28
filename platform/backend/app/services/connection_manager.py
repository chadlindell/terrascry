"""WebSocket connection manager for real-time event broadcasting."""

from datetime import datetime, timezone

from fastapi import WebSocket


class ConnectionManager:
    """Manage WebSocket connections organized by channel."""

    def __init__(self) -> None:
        self._channels: dict[str, set[WebSocket]] = {}

    async def connect(self, channel: str, websocket: WebSocket) -> None:
        await websocket.accept()
        if channel not in self._channels:
            self._channels[channel] = set()
        self._channels[channel].add(websocket)

    def disconnect(self, channel: str, websocket: WebSocket) -> None:
        if channel in self._channels:
            self._channels[channel].discard(websocket)
            if not self._channels[channel]:
                del self._channels[channel]

    async def broadcast(self, channel: str, event_type: str, payload: dict) -> None:
        """Broadcast a JSON envelope to all connections on a channel."""
        message = {
            "type": event_type,
            "payload": payload,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if channel not in self._channels:
            return
        dead: list[WebSocket] = []
        for ws in self._channels[channel]:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._channels[channel].discard(ws)


manager = ConnectionManager()
