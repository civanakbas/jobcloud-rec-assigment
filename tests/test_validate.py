import joblib
import pytest

from mlops_assignment.config import SCHEMA_VERSION
from mlops_assignment.validate import MODEL_METRIC, validate_candidate


def _write_bundle(path, mae: float, schema_version: int = SCHEMA_VERSION) -> None:
    joblib.dump(
        {
            "schema_version": schema_version,
            "metrics": [{"model": MODEL_METRIC, "mae": mae}],
        },
        path,
    )


def test_candidate_within_regression_budget_passes(tmp_path):
    candidate = tmp_path / "candidate.joblib"
    incumbent = tmp_path / "incumbent.joblib"
    _write_bundle(candidate, 104.0)
    _write_bundle(incumbent, 100.0)

    result = validate_candidate(str(candidate), str(incumbent), max_regression=0.05)

    assert result == {"candidate_mae": 104.0, "incumbent_mae": 100.0}


def test_regressed_candidate_is_rejected(tmp_path):
    candidate = tmp_path / "candidate.joblib"
    incumbent = tmp_path / "incumbent.joblib"
    _write_bundle(candidate, 106.0)
    _write_bundle(incumbent, 100.0)

    with pytest.raises(ValueError, match="exceeds promotion limit"):
        validate_candidate(str(candidate), str(incumbent), max_regression=0.05)
