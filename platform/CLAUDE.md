# TERRASCRY Web Platform

Sub-project conventions for the web-based survey visualization platform.

## Project Structure

```
platform/
├── backend/           # FastAPI REST API
│   ├── app/
│   │   ├── main.py           # FastAPI app, CORS, health endpoint
│   │   ├── config.py         # Pydantic Settings (env prefix: TERRASCRY_)
│   │   ├── api_models.py     # Scenario Pydantic models
│   │   ├── dataset.py        # Dataset, GridData, SurveyPoint models
│   │   ├── routers/
│   │   │   ├── scenarios.py  # GET /api/scenarios, /api/scenarios/{name}
│   │   │   ├── surveys.py    # POST /api/surveys/simulate
│   │   │   └── datasets.py   # GET /api/datasets, /api/datasets/{id}
│   │   └── services/
│   │       ├── physics.py    # PhysicsEngine wrapping geosim
│   │       └── dataset_store.py  # Filesystem JSON CRUD
│   └── tests/
├── frontend/          # Vite + React + TypeScript
│   └── src/
│       ├── api.ts            # Typed fetch wrappers
│       ├── colormap.ts       # Viridis/plasma/inferno colormaps
│       ├── components/
│       │   ├── AppShell.tsx       # CSS Grid layout (320px sidebar + main)
│       │   ├── SplitWorkspace.tsx # Resizable 2D/3D split view
│       │   ├── ScenarioSelector.tsx
│       │   ├── RunSurveyButton.tsx
│       │   ├── ColorScaleControl.tsx
│       │   ├── MapView.tsx        # deck.gl 2D heatmap
│       │   ├── SceneView.tsx      # R3F 3D scene
│       │   ├── TerrainMesh.tsx
│       │   ├── BuriedObjects.tsx
│       │   └── SurveyPath3D.tsx
│       ├── hooks/
│       │   ├── useScenarios.ts    # TanStack Query hooks
│       │   ├── useSimulate.ts     # Mutation hook
│       │   └── useDataset.ts      # Active dataset management
│       └── stores/
│           ├── appStore.ts        # Selection, sidebar, view mode
│           └── colorScaleStore.ts # Colormap, range (shared between views)
└── Makefile
```

## Commands

### Backend
```bash
cd platform/backend && pip install -e ".[dev]"   # Install for development
cd platform/backend && pytest tests/ -v           # Run tests
cd platform/backend && ruff check app/            # Lint
cd platform/backend && uvicorn app.main:app --reload  # Dev server (port 8000)
```

### Frontend
```bash
cd platform/frontend && npm install               # Install dependencies
cd platform/frontend && npm run dev               # Dev server (port 5173)
cd platform/frontend && npm run build             # Production build
cd platform/frontend && npx tsc --noEmit          # Type check
```

### Both (via Makefile)
```bash
cd platform && make dev     # Start backend + frontend
cd platform && make test    # Run all tests
cd platform && make lint    # Lint backend + frontend
cd platform && make build   # Production build
```

## Tech Stack

### Backend
- **FastAPI** — REST API framework
- **Pydantic** — Request/response models, settings
- **geosim** — Internal physics engine (dipole magnetics, gradiometer simulation)
- **pytest + httpx** — Async test client

### Frontend
- **Vite + React 19 + TypeScript** — Build toolchain
- **Tailwind CSS v4** — Utility-first styling, zinc dark theme
- **Zustand** — Lightweight state (selection, view mode)
- **TanStack Query** — Server state, caching, mutations
- **deck.gl** — 2D heatmap (BitmapLayer + OrthographicView)
- **React Three Fiber + drei** — 3D scene visualization
- **react-resizable-panels** — Split view layout

## Architecture Decisions

- **GridData** uses flat row-major `values[]` in nanoTesla for direct WebGL TypedArray mapping
- **Dataset payloads** stored in TanStack Query cache, not Zustand (avoids React state overhead with large arrays)
- **BitmapLayer** for heatmap (not HeatmapLayer) — pre-gridded data should not be re-processed with KDE
- **OrthographicView** for 2D — local meter coordinates, not lat/lon
- **camera.up = [0,0,1]** for 3D — matches Z-up coordinate convention
- **Vite proxy** routes `/api` to backend at localhost:8000

## Conventions

- Backend follows root CLAUDE.md ruff config (line-length=100, Python 3.10+)
- Frontend uses TypeScript strict mode
- Component files are PascalCase, hooks use `use` prefix
- All API types mirror backend Pydantic models exactly
