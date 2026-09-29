"""Publish tested images or verify their CI evidence before deployment."""

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import fire

from mlops_assignment.delivery import common


def verify_release(run: dict, release: dict, repository: str, digest: str, run_id: int) -> str:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("Expected a full sha256 image digest")
    if not (
        run.get("id") == run_id
        and run.get("path") == ".github/workflows/ci.yml"
        and run.get("status") == "completed"
        and run.get("conclusion") == "success"
        and run.get("event") == "push"
        and run.get("head_branch") == "main"
        and run.get("head_repository", {}).get("full_name") == repository
    ):
        raise ValueError(
            "Deployment requires a successful main-branch CI push run in this repository"
        )
    image = f"ghcr.io/{repository.lower()}@{digest}"
    if not (
        release.get("image") == image
        and release.get("repository") == repository
        and release.get("sha") == run.get("head_sha")
        and isinstance(release.get("sha"), str)
        and re.fullmatch(r"[0-9a-f]{40}", release["sha"])
        and release.get("run_id") == run_id
        and release.get("run_attempt") == run.get("run_attempt")
    ):
        raise ValueError("Image digest or provenance does not match the successful CI run")
    return image


def publish() -> None:
    """Publish the saved tested image; record its registry digest and CI identity."""
    repository = common.repository()
    commit = common.required("GITHUB_SHA")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Expected a full commit SHA")
    run_id = common.positive_id("GITHUB_RUN_ID")
    attempt = common.positive_id("GITHUB_RUN_ATTEMPT")
    common.required("GITHUB_STEP_SUMMARY")
    image_repository = "ghcr.io/" + repository.lower()
    tag = f"{image_repository}:sha-{commit}"
    common.command("docker", "load", "--input", "tested-image.tar")
    common.command("docker", "tag", "mlops-assignment:ci", tag)
    common.command("docker", "push", tag)
    inspection = json.loads(common.command("docker", "image", "inspect", tag))
    references = [
        value
        for value in inspection[0].get("RepoDigests", [])
        if value.startswith(image_repository + "@")
    ]
    if len(references) != 1:
        raise ValueError("Registry did not return exactly one expected image digest")
    image = references[0]
    common.digest(image.split("@", 1)[1])
    release = dict(
        image=image, sha=commit, run_id=run_id, run_attempt=attempt, repository=repository
    )
    Path("release.json").write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    common.summary(f"Verified image: `{image}`\n\nCommit: `{commit}`")


def verify() -> None:
    """Fetch and validate CI evidence, then write the verified image to Actions output."""
    repository = common.repository()
    run_id = common.positive_id("CI_RUN_ID")
    digest = common.digest(common.required("IMAGE_DIGEST"))
    common.required("GITHUB_OUTPUT")
    run = json.loads(common.command("gh", "api", f"repos/{repository}/actions/runs/{run_id}"))
    attempt = run.get("run_attempt")
    if type(attempt) is not int or attempt < 1:
        raise ValueError("Invalid CI run attempt")
    with TemporaryDirectory() as directory:
        common.command(
            "gh",
            "run",
            "download",
            str(run_id),
            "--repo",
            repository,
            "--name",
            f"release-manifest-{attempt}",
            "--dir",
            directory,
        )
        release = json.loads((Path(directory) / "release.json").read_text(encoding="utf-8"))
    image = verify_release(run, release, repository, digest, run_id)
    common.output("image", image)


if __name__ == "__main__":
    fire.Fire({"publish": publish, "verify": verify})
