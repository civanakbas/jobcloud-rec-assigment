"""Validate a candidate model bundle before promotion."""

from __future__ import annotations

import math
from typing import Any

import fire
import joblib

from mlops_assignment.config import SCHEMA_VERSION

MODEL_METRIC = "HGB on views"


def _mae(bundle: dict[str, Any]) -> float:
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("candidate schema version is incompatible")

    metric = next(
        (item for item in bundle.get("metrics", []) if item.get("model") == MODEL_METRIC),
        None,
    )
    if metric is None or not math.isfinite(float(metric.get("mae", math.nan))):
        raise ValueError(f"candidate is missing a finite '{MODEL_METRIC}' MAE")
    return float(metric["mae"])


def validate_candidate(
    candidate_path: str,
    incumbent_path: str | None = None,
    max_regression: float = 0.05,
) -> dict[str, float | None]:
    """Validate schema and reject excessive MAE regression against an incumbent."""
    if max_regression < 0:
        raise ValueError("max_regression must be non-negative")

    candidate_mae = _mae(joblib.load(candidate_path))
    incumbent_mae = None
    if incumbent_path:
        incumbent_mae = _mae(joblib.load(incumbent_path))
        limit = incumbent_mae * (1 + max_regression)
        if candidate_mae > limit:
            raise ValueError(
                f"candidate MAE {candidate_mae:.4f} exceeds promotion limit {limit:.4f}"
            )

    return {"candidate_mae": candidate_mae, "incumbent_mae": incumbent_mae}


def validate_command(
    candidate: str, incumbent: str | None = None, max_regression: float = 0.05
) -> None:
    """Validate a candidate artifact against an optional incumbent."""
    print(validate_candidate(candidate, incumbent, max_regression))


def main() -> None:
    fire.Fire(validate_command)


if __name__ == "__main__":
    main()
