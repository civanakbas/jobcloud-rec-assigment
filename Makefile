UV ?= uv
RUN = $(UV) run --locked --offline

.PHONY: help sync lint format format-check typecheck check test up down logs

help:
	@echo "sync          Install locked runtime and development dependencies"
	@echo "lint          Check Python code with Ruff"
	@echo "format        Format Python code with Ruff (writes files)"
	@echo "format-check  Check formatting without changing files"
	@echo "typecheck     Check Python types with ty"
	@echo "check         Run lint, format-check, and typecheck"
	@echo "test          Run the offline pytest suite"
	@echo "up            Build and start the API with Docker Compose"
	@echo "down          Stop and remove the Compose services"
	@echo "logs          Follow API container logs"

sync:
	$(UV) sync --locked

lint:
	$(RUN) ruff check . --fix

format:
	$(RUN) ruff format .

format-check:
	$(RUN) ruff format --check .

typecheck:
	$(RUN) ty check

check: lint format-check typecheck

test:
	$(RUN) python -m pytest -q

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs --follow api
