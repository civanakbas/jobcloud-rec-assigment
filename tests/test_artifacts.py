import pathlib

import pytest

from mlops_assignment import s3_utils
from mlops_assignment.train import train


class FakeS3Client:
    def __init__(self) -> None:
        self.uploaded: tuple[str, str, str] | None = None

    def upload_file(self, local_path: str, bucket: str, key: str) -> None:
        self.uploaded = (local_path, bucket, key)

    def download_file(self, bucket: str, key: str, local_path: str) -> None:
        pathlib.Path(local_path).write_bytes(f"{bucket}/{key}".encode())


def test_s3_artifact_round_trip_uses_requested_location(tmp_path, monkeypatch):
    client = FakeS3Client()
    monkeypatch.setattr(s3_utils, "_client", lambda region: client)
    source = tmp_path / "source.joblib"
    source.write_bytes(b"model")

    uri = s3_utils.upload_artifact(str(source), "model-bucket", "models/v1.joblib")
    destination = tmp_path / "nested" / "model.joblib"
    result = s3_utils.download_artifact("model-bucket", "models/v1.joblib", str(destination))

    assert uri == "s3://model-bucket/models/v1.joblib"
    assert client.uploaded == (str(source), "model-bucket", "models/v1.joblib")
    assert result == destination
    assert destination.read_bytes() == b"model-bucket/models/v1.joblib"


def test_training_rejects_partial_s3_configuration_before_reading_data():
    with pytest.raises(ValueError, match="must be provided together"):
        train("missing.csv", "missing.csv", s3_bucket="model-bucket")
