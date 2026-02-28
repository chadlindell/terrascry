"""Streaming service — MQTT bridge or simulated survey data via WebSocket.

Two modes controlled by settings.mqtt_broker_host:
- Simulation mode (default, empty host): walks a survey path point-by-point
  using PhysicsEngine, broadcasts stream_point and anomaly_detected events.
- MQTT mode (when host set): subscribes to MQTT topics, relays to WebSocket.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import numpy as np
from geosim.magnetics.dipole import gradiometer_reading
from geosim.streaming.anomaly_detector import SimpleAnomalyDetector

from app.config import settings
from app.services.connection_manager import manager
from app.services.physics import SENSOR_HEIGHT, SENSOR_SEPARATION, PhysicsEngine

logger = logging.getLogger(__name__)

STREAM_CHANNEL = "stream"


class StreamingService:
    """Manages streaming data to WebSocket clients via simulation or MQTT."""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._running = False
        self._mode: str = "simulation"
        self._scenario_name: str = ""
        self._points_sent: int = 0
        self._anomalies_detected: int = 0
        self._engine = PhysicsEngine()
        self._detector = SimpleAnomalyDetector()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def mode(self) -> str:
        return self._mode

    def status(self) -> dict:
        return {
            "running": self._running,
            "mode": self._mode,
            "scenario_name": self._scenario_name,
            "points_sent": self._points_sent,
            "anomalies_detected": self._anomalies_detected,
        }

    async def start(self, scenario_name: str = "") -> None:
        """Start streaming. Uses MQTT if broker configured, else simulation."""
        if self._running:
            return

        self._points_sent = 0
        self._anomalies_detected = 0
        self._detector.reset()

        if settings.mqtt_broker_host:
            self._mode = "mqtt"
            self._scenario_name = ""
            self._task = asyncio.create_task(self._mqtt_loop())
        else:
            self._mode = "simulation"
            self._scenario_name = scenario_name or "single-ferrous-target"
            # Validate scenario exists before launching background task
            self._engine.load_scenario(self._scenario_name)
            self._task = asyncio.create_task(self._simulation_loop())

        self._running = True

    async def stop(self) -> None:
        """Stop streaming."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _simulation_loop(self) -> None:
        """Walk a survey path point-by-point, broadcasting each reading."""
        try:
            scenario = self._engine.load_scenario(self._scenario_name)
            path = self._engine.generate_path(
                scenario, line_spacing=1.0, sample_spacing=0.5
            )
            sources = scenario.magnetic_sources

            rate_hz = settings.stream_rate_hz
            interval = 1.0 / rate_hz

            for i in range(len(path)):
                if not self._running:
                    break

                pos = path[i]
                x, y = float(pos[0]), float(pos[1])

                # Compute gradient at this point
                if sources:
                    obs = np.array([[x, y, SENSOR_HEIGHT]])
                    _, _, grad = gradiometer_reading(
                        obs, sources, SENSOR_SEPARATION
                    )
                    gradient_nt = float(grad[0] * 1e9)
                else:
                    gradient_nt = 0.0

                now = datetime.now(timezone.utc).isoformat()

                # Broadcast stream point
                await manager.broadcast(STREAM_CHANNEL, "stream_point", {
                    "x": x,
                    "y": y,
                    "gradient_nt": gradient_nt,
                    "timestamp": now,
                })
                self._points_sent += 1

                # Check for anomaly
                result = self._detector.process_sample(gradient_nt)
                if result["is_anomaly"]:
                    self._anomalies_detected += 1
                    await manager.broadcast(STREAM_CHANNEL, "anomaly_detected", {
                        "x": x,
                        "y": y,
                        "anomaly_strength_nt": result["residual_nt"],
                        "anomaly_type": "magnetic",
                        "confidence": min(result["sigma"] / 5.0, 1.0),
                        "timestamp": now,
                    })

                await asyncio.sleep(interval)

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Simulation loop error")
        finally:
            self._running = False

    async def _mqtt_loop(self) -> None:
        """Subscribe to MQTT topics and relay messages to WebSocket clients."""
        try:
            import aiomqtt
        except ImportError:
            logger.error("aiomqtt not installed; cannot use MQTT mode")
            self._running = False
            return

        backoff = 1.0
        max_backoff = 30.0

        while self._running:
            try:
                async with aiomqtt.Client(
                    hostname=settings.mqtt_broker_host,
                    port=settings.mqtt_broker_port,
                ) as client:
                    await client.subscribe("terrascry/pathfinder/data/corrected")
                    await client.subscribe("terrascry/pathfinder/anomaly/detected")
                    backoff = 1.0  # Reset on successful connect

                    async for message in client.messages:
                        if not self._running:
                            break
                        try:
                            import json
                            payload = json.loads(message.payload.decode())
                            topic = str(message.topic)

                            if "data/corrected" in topic:
                                await manager.broadcast(
                                    STREAM_CHANNEL, "stream_point", payload
                                )
                                self._points_sent += 1
                            elif "anomaly/detected" in topic:
                                await manager.broadcast(
                                    STREAM_CHANNEL, "anomaly_detected", payload
                                )
                                self._anomalies_detected += 1
                        except Exception:
                            logger.exception("Error processing MQTT message")

            except asyncio.CancelledError:
                break
            except Exception:
                if not self._running:
                    break
                logger.warning(
                    "MQTT connection failed, retrying in %.1fs", backoff
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

        self._running = False


streaming_service = StreamingService()
