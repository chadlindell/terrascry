"""Centralized configuration via pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Platform configuration, populated from environment variables with TERRASCRY_ prefix."""

    model_config = {"env_prefix": "TERRASCRY_"}

    scenarios_dir: Path = Path(__file__).resolve().parents[3] / "geosim" / "scenarios"
    data_dir: Path = Path.home() / ".terrascry" / "datasets"
    cors_origins: list[str] = ["http://localhost:5173"]

    # MQTT streaming (empty host = simulation mode)
    mqtt_broker_host: str = ""
    mqtt_broker_port: int = 1883
    stream_rate_hz: float = 10.0


settings = Settings()
