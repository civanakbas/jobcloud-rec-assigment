"""Manual candidate preparation and upload; never promotes or restarts serving."""

from dataclasses import dataclass
from pathlib import Path

import fire

from mlops_assignment.delivery import common
from mlops_assignment.delivery.metadata import write_metadata
from mlops_assignment.train import train as train_model
from mlops_assignment.validate import validate_candidate

POSTINGS = Path("data/postings.csv")
COMPANIES = Path("data/companies/company_industries.csv")
ARTIFACT = Path("artifacts/candidate.joblib")
METADATA = Path("artifacts/candidate.json")


@dataclass(frozen=True)
class Snapshot:
    bucket: str
    postings_version: str
    companies_version: str

    @classmethod
    def from_env(cls) -> "Snapshot":
        settings = cls(
            common.required("DATA_BUCKET"),
            common.required("POSTINGS_VERSION"),
            common.required("COMPANIES_VERSION"),
        )
        if "null" in (settings.postings_version, settings.companies_version):
            raise ValueError("Explicit non-null S3 object versions are required")
        return settings


def validate() -> None:
    """Reject incomplete snapshot selection before AWS credentials are acquired."""
    Snapshot.from_env()


def download_snapshot(s3, snapshot: Snapshot) -> None:
    for key, version, destination in (
        ("postings.csv", snapshot.postings_version, POSTINGS),
        ("companies/company_industries.csv", snapshot.companies_version, COMPANIES),
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(snapshot.bucket, key, str(destination), ExtraArgs={"VersionId": version})


def download() -> None:
    snapshot = Snapshot.from_env()
    download_snapshot(common.client("s3"), snapshot)


def train() -> None:
    train_model(str(POSTINGS), str(COMPANIES), str(ARTIFACT))
    validate_candidate(str(ARTIFACT))


def upload_files(s3, bucket: str, prefix: str) -> None:
    for path, filename, content_type in (
        (ARTIFACT, "model.joblib", "application/octet-stream"),
        (METADATA, "metadata.json", "application/json"),
    ):
        with path.open("rb") as body:
            s3.put_object(
                Bucket=bucket,
                Key=f"{prefix}/{filename}",
                Body=body,
                ContentType=content_type,
                IfNoneMatch="*",
            )


def upload() -> None:
    snapshot = Snapshot.from_env()
    bucket = common.required("MODEL_BUCKET")
    run_id = common.positive_id("GITHUB_RUN_ID")
    attempt = common.positive_id("GITHUB_RUN_ATTEMPT")
    commit = common.required("GITHUB_SHA")
    common.required("GITHUB_STEP_SUMMARY")
    # Derive the prefix instead of accepting arbitrary production paths from env.
    prefix = f"models/views/candidates/{run_id}/{attempt}"
    write_metadata(
        artifact=str(ARTIFACT),
        postings=str(POSTINGS),
        companies=str(COMPANIES),
        postings_version=snapshot.postings_version,
        companies_version=snapshot.companies_version,
        bucket=snapshot.bucket,
        commit=commit,
        run_id=str(run_id),
        run_attempt=str(attempt),
        out=str(METADATA),
    )
    upload_files(common.client("s3"), bucket, prefix)
    common.summary(
        f"Candidate only: `s3://{bucket}/{prefix}/`\n\nNo model promotion or service restart performed."
    )


if __name__ == "__main__":
    fire.Fire({"validate": validate, "download": download, "train": train, "upload": upload})
