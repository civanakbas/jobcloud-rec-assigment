"""FastAPI serving layer for the views-prediction model."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from mlops_assignment import __version__
from mlops_assignment.config import DEFAULT_ARTIFACT_PATH
from mlops_assignment.predict import load_bundle, predict_views
from mlops_assignment.s3_utils import download_artifact

ARTIFACT_PATH = os.environ.get("VIEWS_MODEL_ARTIFACT", DEFAULT_ARTIFACT_PATH)
S3_BUCKET = os.environ.get("VIEWS_MODEL_S3_BUCKET")
S3_KEY = os.environ.get("VIEWS_MODEL_S3_KEY")
S3_REGION = os.environ.get("AWS_REGION", "us-east-1")


def _prepare_model() -> None:
    artifact = Path(ARTIFACT_PATH)
    if not artifact.is_file() and S3_BUCKET and S3_KEY:
        download_artifact(S3_BUCKET, S3_KEY, ARTIFACT_PATH, S3_REGION)
    load_bundle(ARTIFACT_PATH)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _prepare_model()
    yield


app = FastAPI(
    title="Job-posting views predictor",
    version=__version__,
    lifespan=lifespan,
)

REQUEST_COUNT = Counter(
    "views_model_http_requests_total",
    "HTTP requests handled by the prediction service.",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "views_model_http_request_duration_seconds",
    "HTTP request latency for the prediction service.",
    ["method", "path"],
)


class Posting(BaseModel):
    """Minimal raw-posting schema the model needs. Optional fields can be null."""

    job_id: Optional[int] = None
    company_id: Optional[float] = None
    title: Optional[str] = ""
    description: Optional[str] = ""
    skills_desc: Optional[str] = None
    normalized_salary: Optional[float] = None
    remote_allowed: Optional[float] = None
    sponsored: Optional[float] = None
    formatted_work_type: Optional[str] = None
    formatted_experience_level: Optional[str] = None
    pay_period: Optional[str] = None
    application_type: Optional[str] = None
    original_listed_time: float = Field(..., description="Epoch ms when first listed.")
    listed_time: float = Field(..., description="Epoch ms when (re)listed.")
    expiry: float = Field(..., description="Epoch ms when the posting expires.")


class PredictRequest(BaseModel):
    postings: list[Posting]
    target: Literal["views", "views_per_day"] = "views"


class PredictResponse(BaseModel):
    target: str
    predictions: list[float]


@app.middleware("http")
async def observe_request(request: Request, call_next):
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        # Routing has completed by this point. Never use user-controlled URL paths
        # or arbitrary HTTP method strings as metric labels.
        route = request.scope.get("route")
        path = getattr(route, "path", "__unmatched__")
        method = (
            request.method
            if request.method
            in {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"}
            else "OTHER"
        )
        REQUEST_COUNT.labels(method, path, str(status)).inc()
        REQUEST_LATENCY.labels(method, path).observe(time.perf_counter() - started)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/ready")
def ready() -> dict[str, str]:
    try:
        load_bundle(ARTIFACT_PATH)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="model artifact unavailable") from exc
    return {"status": "ready", "version": __version__}


@app.get("/metrics", response_class=Response)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model-info")
def model_info() -> dict:
    bundle = load_bundle(ARTIFACT_PATH)
    return {
        "metrics": bundle.get("metrics", []),
        "metadata": bundle.get("metadata", {}),
        "schema_version": bundle["schema_version"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if not req.postings:
        raise HTTPException(status_code=400, detail="`postings` must be non-empty")
    bundle = load_bundle(ARTIFACT_PATH)
    raw = pd.DataFrame([p.model_dump() for p in req.postings])
    preds = predict_views(raw, bundle, target=req.target)
    return PredictResponse(target=req.target, predictions=[float(x) for x in preds])
