import math

import joblib
import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, Counter, Histogram

from mlops_assignment import api
from mlops_assignment.config import SCHEMA_VERSION
from mlops_assignment.train import train
from tests.support import sample_data


@pytest.fixture(scope="module")
def trained_artifact(tmp_path_factory):
    directory = tmp_path_factory.mktemp("api-model")
    postings, companies = sample_data()
    postings.to_csv(directory / "postings.csv", index=False)
    companies.to_csv(directory / "companies.csv", index=False)
    path = directory / "model.joblib"
    train(str(directory / "postings.csv"), str(directory / "companies.csv"), str(path))
    return path


@pytest.fixture
def client(monkeypatch, trained_artifact):
    monkeypatch.setattr(api, "ARTIFACT_PATH", str(trained_artifact))
    monkeypatch.setattr(api, "S3_BUCKET", None)
    load_bundle = api.load_bundle
    load_bundle.cache_clear()
    with TestClient(api.app) as client:
        yield client
    load_bundle.cache_clear()


@pytest.fixture
def posting():
    return {
        "company_id": 1,
        "title": "Data Engineer",
        "description": "Python and SQL",
        "normalized_salary": 120000,
        "remote_allowed": 1,
        "sponsored": 0,
        "formatted_work_type": "Full-time",
        "formatted_experience_level": "Mid-Senior level",
        "pay_period": "YEARLY",
        "application_type": "OffsiteApply",
        "original_listed_time": 1713000000000,
        "listed_time": 1713000000000,
        "expiry": 1715592000000,
    }


def test_health_readiness_metadata_and_metrics(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["status"] == "ready"
    info = client.get("/model-info")
    assert info.status_code == 200
    assert info.json()["schema_version"] == SCHEMA_VERSION
    assert info.json()["metadata"]["training_rows"] > 0
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "views_model_http_requests_total" in response.text


@pytest.mark.parametrize("target", ["views", "views_per_day"])
def test_prediction_over_http(client, posting, target):
    response = client.post("/predict", json={"postings": [posting, posting], "target": target})
    assert response.status_code == 200
    body = response.json()
    assert body["target"] == target
    assert len(body["predictions"]) == 2
    assert all(math.isfinite(value) and value >= 0 for value in body["predictions"])


@pytest.mark.parametrize(
    "body,status",
    [
        ({"postings": []}, 400),
        ({"postings": [{}]}, 422),
        ({"postings": [], "target": "unknown"}, 422),
        ({}, 422),
    ],
)
def test_invalid_requests(client, body, status):
    assert client.post("/predict", json=body).status_code == status


def test_readiness_failure_keeps_liveness(client, monkeypatch):
    def unavailable(_):
        raise FileNotFoundError("internal artifact path")

    monkeypatch.setattr(api, "load_bundle", unavailable)
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"detail": "model artifact unavailable"}
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize("incompatible", [False, True])
def test_startup_rejects_missing_or_incompatible_artifact(tmp_path, monkeypatch, incompatible):
    path = tmp_path / "model.joblib"
    if incompatible:
        joblib.dump({"schema_version": SCHEMA_VERSION + 1}, path)
    monkeypatch.setattr(api, "ARTIFACT_PATH", str(path))
    monkeypatch.setattr(api, "S3_BUCKET", None)
    api.load_bundle.cache_clear()
    with pytest.raises(ValueError if incompatible else FileNotFoundError):
        with TestClient(api.app):
            raise AssertionError("Startup must fail before serving requests")


def test_metric_labels_are_bounded(client, monkeypatch):
    registry = CollectorRegistry()
    count = Counter("test_requests", "Requests", ["method", "path", "status"], registry=registry)
    latency = Histogram("test_latency", "Latency", ["method", "path"], registry=registry)
    monkeypatch.setattr(api, "REQUEST_COUNT", count)
    monkeypatch.setattr(api, "REQUEST_LATENCY", latency)
    for index in range(20):
        assert client.get(f"/unknown-{index}").status_code == 404
        assert client.request(f"CUSTOM{index}", f"/unknown-{index}").status_code == 404
    assert client.get("/health?user=private").status_code == 200
    samples = [sample for metric in registry.collect() for sample in metric.samples]
    assert {s.labels["path"] for s in samples} == {"__unmatched__", "/health"}
    assert {s.labels["method"] for s in samples} == {"GET", "OTHER"}
    totals = [s for s in samples if s.name == "test_requests_total"]
    assert sum(s.value for s in totals if s.labels["path"] == "__unmatched__") == 40
