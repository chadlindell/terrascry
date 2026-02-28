# TERRASCRY Web Platform

Web-based survey simulation and visualization platform for the TERRASCRY multi-sensor geophysical survey system. Wraps the geosim physics engine with a FastAPI backend and React frontend.

## Quick Start

```bash
# Prerequisites: Python 3.10+, Node.js 22+

# 1. Install the physics engine
cd geosim && pip install -e .

# 2. Install the backend
cd platform/backend && pip install -e ".[dev]"

# 3. Install the frontend
cd platform/frontend && npm install

# 4. Start both servers
cd platform && make dev
# Backend: http://localhost:8000
# Frontend: http://localhost:5173
```

## Architecture

```
Browser (React + deck.gl + R3F)
  │  fetch /api/*
  ▼
Vite dev proxy (:5173 → :8000)
  │
  ▼
FastAPI (:8000)
  ├─ /api/scenarios      — List/upload scenario JSON files
  ├─ /api/surveys        — Trigger simulation (single + batch)
  ├─ /api/datasets       — CRUD for stored results
  ├─ /api/datasets/*/export — CSV / ESRI ASCII Grid export
  ├─ /api/ws/{channel}   — WebSocket real-time events
  └─ /api/health         — Health check
  │
  ▼
geosim (physics engine)
  └─ dipole magnetics, gradiometer simulation, grid computation
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/scenarios` | List all scenarios |
| GET | `/api/scenarios/{name}` | Get scenario detail |
| POST | `/api/scenarios` | Upload new scenario (409 on duplicate) |
| POST | `/api/surveys/simulate` | Run single simulation |
| POST | `/api/surveys/batch` | Run batch simulation (max 10) |
| GET | `/api/datasets` | List stored datasets |
| GET | `/api/datasets/{id}` | Get full dataset |
| DELETE | `/api/datasets/{id}` | Delete dataset |
| GET | `/api/datasets/{id}/export` | Export as CSV/grid_csv/asc |
| WS | `/api/ws/{channel}` | WebSocket events |
| GET | `/api/health` | Health check |

Interactive API docs available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc` (ReDoc).

## Testing

```bash
# Backend tests
cd platform/backend && pytest tests/ -v

# Frontend tests
cd platform/frontend && npm test

# Frontend type check
cd platform/frontend && npx tsc --noEmit

# All tests via Makefile
cd platform && make test
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make dev` | Start backend + frontend dev servers |
| `make test` | Run all tests (backend pytest + frontend vitest + typecheck) |
| `make lint` | Lint backend (ruff) + frontend (eslint + typecheck) |
| `make build` | Production frontend build |
| `make clean` | Remove build artifacts |

## Environment Variables

All prefixed with `TERRASCRY_`:

| Variable | Default | Description |
|----------|---------|-------------|
| `TERRASCRY_SCENARIOS_DIR` | `geosim/scenarios` | Path to scenario JSON files |
| `TERRASCRY_DATA_DIR` | `~/.terrascry/datasets` | Path to stored dataset files |
| `TERRASCRY_CORS_ORIGINS` | `["http://localhost:5173"]` | Allowed CORS origins |

## Tech Stack

**Backend:** FastAPI, Pydantic, geosim, pytest + httpx

**Frontend:** Vite, React 19, TypeScript, Tailwind CSS v4, Zustand, TanStack Query, deck.gl, React Three Fiber
