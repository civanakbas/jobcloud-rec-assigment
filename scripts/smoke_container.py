"""Offline HTTP smoke check of an already-built image and a trusted model artifact.

Uses Fire from the project environment on the host. Starts the image's normal command,
executes an HTTP client inside its network-isolated container, and always cleans up.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path

import fire

HTTP_CHECK = r"""
import json
import math
import time
import urllib.error
import urllib.request

base = "http://127.0.0.1:8000"
def request(path, body=None, expected=200):
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=payload, headers={"Content-Type": "application/json"})
    try:
        response = urllib.request.urlopen(req, timeout=5)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        content = response.read().decode()
        assert response.status == expected, (path, response.status, content)
        return content if path == "/metrics" else json.loads(content)

for attempt in range(60):
    try:
        ready = request("/ready")
        assert ready["status"] == "ready"
        break
    except (OSError, AssertionError):
        time.sleep(0.5)
else:
    raise RuntimeError("Service did not become ready within the startup retry budget")
assert request("/health")["status"] == "ok"
assert request("/model-info")["schema_version"] == 1
posting = {
    "company_id": 1234, "title": "Data Engineer",
    "description": "Build reliable data pipelines using Python and SQL.",
    "normalized_salary": 120000, "remote_allowed": 1, "sponsored": 0,
    "formatted_work_type": "Full-time", "formatted_experience_level": "Mid-Senior level",
    "pay_period": "YEARLY", "application_type": "OffsiteApply",
    "original_listed_time": 1713000000000, "listed_time": 1713000000000,
    "expiry": 1715592000000,
}
for target in ("views", "views_per_day"):
    result = request("/predict", {"postings": [posting], "target": target})
    assert result["target"] == target
    assert len(result["predictions"]) == 1
    assert all(math.isfinite(x) and x >= 0 for x in result["predictions"])
    print(json.dumps(result))
request("/predict", {"postings": []}, 400)
request("/predict", {"postings": [{}]}, 422)
request("/predict", {"postings": [posting], "target": "invalid"}, 422)
for index in range(3):
    request(f"/unknown-{index}", expected=404)
metrics = request("/metrics")
assert 'path="__unmatched__"' in metrics
assert '/unknown-' not in metrics
assert 'path="/predict"' in metrics
assert 'views_model_http_request_duration_seconds_bucket' in metrics
print("PASS: health, readiness, metadata, both predictions, invalid requests, bounded metrics")
"""


def smoke(
    image: str = "mlops-assignment:local", artifact: str = "artifacts/views_baseline.joblib"
) -> None:
    """Verify an existing image through Compose without building, pulling or publishing ports."""
    artifact_path = Path(artifact).resolve(strict=True)
    if not artifact_path.is_file():
        raise ValueError("The model artifact must be a file")
    root = Path(__file__).resolve().parents[1]
    project = "mlops-smoke-" + uuid.uuid4().hex
    container = project + "-api"
    environment = {**os.environ, "API_IMAGE": image, "MODEL_ARTIFACT": str(artifact_path)}
    compose = [
        "docker",
        "compose",
        "--project-name",
        project,
        "--file",
        str(root / "compose.yaml"),
        "--file",
        str(root / "compose.smoke.yaml"),
    ]
    try:
        subprocess.run(
            [
                *compose,
                "run",
                "--detach",
                "--no-deps",
                "--pull",
                "never",
                "--name",
                container,
                "api",
            ],
            env=environment,
            check=True,
            timeout=60,
        )
        subprocess.run(
            ["docker", "exec", "-i", container, "python", "-"],
            input=HTTP_CHECK,
            text=True,
            check=True,
            timeout=90,
        )
    except BaseException:
        # Diagnostic failure must not hide the original startup/HTTP failure.
        try:
            subprocess.run(["docker", "logs", container], check=False, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass
        raise
    finally:
        subprocess.run(
            [*compose, "down", "--remove-orphans", "--timeout", "5"],
            env=environment,
            check=True,
            timeout=30,
            stdout=subprocess.DEVNULL,
        )


def main() -> None:
    fire.Fire(smoke)


if __name__ == "__main__":
    main()
