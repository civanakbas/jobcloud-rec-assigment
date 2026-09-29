"""Write candidate provenance alongside a trusted, locally trained artifact."""

import hashlib
import json
import re
from pathlib import Path

import fire

from mlops_assignment.predict import load_bundle


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_metadata(
    artifact: str,
    postings: str,
    companies: str,
    postings_version: str,
    companies_version: str,
    bucket: str,
    commit: str,
    run_id: str,
    run_attempt: str,
    out: str,
    lockfile: str = "uv.lock",
) -> None:
    """Record exact data inputs and artifact hash; this is not a promotion decision."""
    if (
        not postings_version
        or not companies_version
        or "null" in (postings_version, companies_version)
    ):
        raise ValueError("Explicit non-null S3 object versions are required")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Expected full training-code commit SHA")
    bundle = load_bundle(artifact)
    document = {
        "status": "candidate_only",
        "code_sha": commit,
        "run_id": str(run_id),
        "run_attempt": str(run_attempt),
        "artifact_sha256": sha256(Path(artifact)),
        "uv_lock_sha256": sha256(Path(lockfile)),
        "schema_version": bundle["schema_version"],
        "training_metadata": bundle["metadata"],
        "metrics": bundle["metrics"],
        "data": [
            {
                "bucket": bucket,
                "key": "postings.csv",
                "version_id": postings_version,
                "sha256": sha256(Path(postings)),
            },
            {
                "bucket": bucket,
                "key": "companies/company_industries.csv",
                "version_id": companies_version,
                "sha256": sha256(Path(companies)),
            },
        ],
        "evaluation": "Random holdout from this training snapshot; not comparable to an incumbent's stored metrics and not authorization to promote.",
    }
    Path(out).write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    fire.Fire(write_metadata)
