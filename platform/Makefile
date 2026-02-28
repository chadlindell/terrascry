.PHONY: help dev test lint build clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

dev: ## Start backend and frontend dev servers
	@echo "Starting TERRASCRY platform..."
	@(cd backend && uvicorn app.main:app --reload --port 8000) & \
	(cd frontend && npm run dev) & \
	wait

test: ## Run all tests
	cd backend && pytest tests/ -v --tb=short
	cd frontend && npx tsc --noEmit

lint: ## Lint backend and frontend
	cd backend && ruff check app/
	cd frontend && npx tsc --noEmit

build: ## Production build
	cd frontend && npm run build

clean: ## Remove build artifacts
	rm -rf frontend/dist frontend/node_modules/.vite
