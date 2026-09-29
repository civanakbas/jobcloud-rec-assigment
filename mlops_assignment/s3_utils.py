"""S3 model artifact transfer helpers."""

from __future__ import annotations

import os
import pathlib
import tempfile


def _client(region: str):
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("S3 support requires the 'aws' extra: pip install -e '.[aws]'") from exc
    return boto3.client("s3", region_name=region)


def upload_artifact(
    local_path: str,
    bucket: str,
    key: str,
    region: str = "us-east-1",
) -> str:
    """Upload an artifact and return its S3 URI."""
    artifact = pathlib.Path(local_path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Artifact not found: {artifact}")

    _client(region).upload_file(str(artifact), bucket, key)
    return f"s3://{bucket}/{key}"


def download_artifact(
    bucket: str,
    key: str,
    local_path: str,
    region: str = "us-east-1",
) -> pathlib.Path:
    """Download to a temporary file and atomically publish it locally."""
    destination = pathlib.Path(local_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary_path = pathlib.Path(temporary_name)
    try:
        _client(region).download_file(bucket, key, str(temporary_path))
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)

    return destination
