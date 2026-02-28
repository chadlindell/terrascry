"""WebSocket router — real-time event streaming by channel."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.connection_manager import manager

router = APIRouter(tags=["websocket"])


@router.websocket("/api/ws/{channel}")
async def websocket_endpoint(websocket: WebSocket, channel: str) -> None:
    await manager.connect(channel, websocket)
    try:
        while True:
            # Keep connection alive; clients can send pings
            data = await websocket.receive_json()
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(channel, websocket)
