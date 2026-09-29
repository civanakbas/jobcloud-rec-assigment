# syntax=docker/dockerfile:1.6
FROM python:3.12.14-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/usr/local \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy

WORKDIR /app

FROM base AS runtime
# Install locked runtime dependencies into system Python, without a venv.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project --no-cache --python /usr/local/bin/python

# Install the local project normally (not editably); no distribution artifacts to manage.
COPY mlops_assignment ./mlops_assignment
RUN uv sync --locked --no-dev --no-editable --no-cache --python /usr/local/bin/python \
    && rm -rf /app/mlops_assignment /app/build /app/*.egg-info

# Imports must come from site-packages without source directories or a venv.
RUN python -c "import sys, sysconfig, pathlib, mlops_assignment; from mlops_assignment.reports import evaluate; assert sys.prefix == sys.base_prefix; assert pathlib.Path(mlops_assignment.__file__).is_relative_to(sysconfig.get_path('purelib'))" \
    && test ! -d /app/.venv \
    && test ! -d /app/mlops_assignment \
    && test ! -d /app/ai_exercise

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/artifacts \
    && chown -R appuser:appuser /app/artifacts

ENV VIEWS_MODEL_ARTIFACT=/app/artifacts/views_baseline.joblib \
    PORT=8000

USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:8000/ready').status==200 else 1)"

CMD ["sh", "-c", "uvicorn mlops_assignment.api:app --host 0.0.0.0 --port ${PORT}"]
