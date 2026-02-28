"""TERRASCRY platform — FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import (
    anomalies,
    datasets,
    datasets_binary,
    exports,
    imports,
    scenarios,
    streaming,
    surveys,
    ws,
)

app = FastAPI(
    title="TERRASCRY Platform",
    version="0.1.0",
    description=(
        "Web platform for TERRASCRY multi-sensor geophysical survey simulation "
        "and visualization. Wraps the geosim physics engine to generate synthetic "
        "gradiometer survey data over configurable buried-object scenarios."
    ),
)
app.include_router(scenarios.router)
app.include_router(surveys.router)
app.include_router(datasets.router)
app.include_router(anomalies.router)
app.include_router(datasets_binary.router)
app.include_router(exports.router)
app.include_router(imports.router)
app.include_router(streaming.router)
app.include_router(ws.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "version": app.version}
