"""Exercise deployment evidence checks without GitHub credentials or AWS."""

import json
from copy import deepcopy

import joblib
import pytest

from mlops_assignment.config import SCHEMA_VERSION
from mlops_assignment.delivery.metadata import write_metadata
from mlops_assignment.delivery.release import verify_release

DIGEST = "sha256:" + "a" * 64
REPO = "Owner/Repo"
RUN = {
    "id": 123,
    "run_attempt": 2,
    "path": ".github/workflows/ci.yml",
    "status": "completed",
    "conclusion": "success",
    "event": "push",
    "head_branch": "main",
    "head_sha": "b" * 40,
    "head_repository": {"full_name": REPO},
}
RELEASE = {
    "image": "ghcr.io/owner/repo@" + DIGEST,
    "sha": "b" * 40,
    "run_id": 123,
    "run_attempt": 2,
    "repository": REPO,
}


def test_matching_ci_evidence_accepts_digest():
    assert verify_release(RUN, RELEASE, REPO, DIGEST, 123) == RELEASE["image"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("conclusion", "failure"),
        ("status", "in_progress"),
        ("event", "pull_request"),
        ("head_branch", "feature"),
        ("path", ".github/workflows/other.yml"),
        ("head_repository", {"full_name": "attacker/fork"}),
        ("id", 456),
    ],
)
def test_untrusted_or_unsuccessful_run_is_rejected(field, value):
    run = deepcopy(RUN)
    run[field] = value
    with pytest.raises(ValueError):
        verify_release(run, RELEASE, REPO, DIGEST, 123)


@pytest.mark.parametrize(
    "field,value",
    [
        ("image", "ghcr.io/attacker/repo@" + DIGEST),
        ("sha", "c" * 40),
        ("run_id", 456),
        ("run_attempt", 1),
        ("repository", "attacker/fork"),
    ],
)
def test_mismatched_manifest_is_rejected(field, value):
    release = {**RELEASE, field: value}
    with pytest.raises(ValueError):
        verify_release(RUN, release, REPO, DIGEST, 123)


@pytest.mark.parametrize(
    "digest", ["latest", "sha256:short", "sha256:" + "c" * 64, DIGEST + "\nimage=evil"]
)
def test_unverified_digest_is_rejected(digest):
    with pytest.raises(ValueError):
        verify_release(RUN, RELEASE, REPO, digest, 123)


def test_candidate_metadata_records_versions_and_hashes(tmp_path):
    artifact = tmp_path / "model.joblib"
    joblib.dump(
        {"schema_version": SCHEMA_VERSION, "metadata": {"training_rows": 10}, "metrics": []},
        artifact,
    )
    postings, companies, lockfile = (
        tmp_path / name for name in ("postings.csv", "companies.csv", "uv.lock")
    )
    for path in (postings, companies, lockfile):
        path.write_text("fixture")
    output = tmp_path / "metadata.json"
    kwargs = dict(
        artifact=str(artifact),
        postings=str(postings),
        companies=str(companies),
        postings_version="version-p",
        companies_version="version-c",
        bucket="data-bucket",
        commit="b" * 40,
        run_id="123",
        run_attempt="2",
        out=str(output),
        lockfile=str(lockfile),
    )
    write_metadata(**kwargs)
    result = json.loads(output.read_text())
    assert result["status"] == "candidate_only"
    assert result["data"][0]["version_id"] == "version-p"
    assert result["data"][1]["version_id"] == "version-c"
    assert len(result["artifact_sha256"]) == 64
    assert result["run_attempt"] == "2"
    with pytest.raises(ValueError, match="non-null"):
        write_metadata(**{**kwargs, "postings_version": "null"})
