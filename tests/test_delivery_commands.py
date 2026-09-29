"""Delivery adapters execute against local mocks/stubs, never live cloud services."""

import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import ANY, Stubber

from mlops_assignment.delivery import candidate, common, deploy, release
from tests.test_delivery import DIGEST, RELEASE, REPO, RUN

SOURCE = "ghcr.io/owner/repo@" + DIGEST
REGISTRY = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
IMAGE = f"{REGISTRY}/views-model@{DIGEST}"
TASK: dict[str, Any] = {
    "family": "views-model",
    "taskDefinitionArn": "old-task",
    "revision": 1,
    "status": "ACTIVE",
    "registeredBy": "owner",
    "requiresAttributes": [],
    "containerDefinitions": [
        {
            "name": "views-model",
            "image": "previous-image",
            "environment": [{"name": "MODEL", "value": "keep"}],
        },
        {"name": "sidecar", "image": "keep-sidecar"},
    ],
    "taskRoleArn": "keep-role",
}


def aws(service):
    return boto3.client(
        service,
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )


@pytest.fixture
def actions_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "GITHUB_REPOSITORY": REPO,
        "GITHUB_SHA": "b" * 40,
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "CI_RUN_ID": "123",
        "IMAGE_DIGEST": DIGEST,
        "DATA_BUCKET": "data-bucket",
        "MODEL_BUCKET": "model-bucket",
        "POSTINGS_VERSION": "v-p",
        "COMPANIES_VERSION": "v-c",
    }.items():
        monkeypatch.setenv(key, value)
    return tmp_path


def test_snapshot_validation_before_any_remote_call(actions_env, monkeypatch):
    monkeypatch.setenv("POSTINGS_VERSION", "null")
    remote = Mock()
    monkeypatch.setattr(common, "client", remote)
    with pytest.raises(ValueError, match="non-null"):
        candidate.download()
    remote.assert_not_called()


def test_download_uses_exact_versions(actions_env):
    s3 = Mock()
    candidate.download_snapshot(s3, candidate.Snapshot.from_env())
    assert s3.download_file.call_args_list[0].kwargs == {"ExtraArgs": {"VersionId": "v-p"}}
    assert s3.download_file.call_args_list[1].kwargs == {"ExtraArgs": {"VersionId": "v-c"}}
    assert Path("data/companies").is_dir()


def test_conditional_upload_uses_supported_sdk_parameters(actions_env):
    candidate.ARTIFACT.parent.mkdir()
    candidate.ARTIFACT.write_bytes(b"model")
    candidate.METADATA.write_text("{}")
    s3 = aws("s3")
    with Stubber(s3) as stub:
        for filename, content_type in (
            ("model.joblib", "application/octet-stream"),
            ("metadata.json", "application/json"),
        ):
            stub.add_response(
                "put_object",
                {},
                {
                    "Bucket": "bucket",
                    "Key": f"prefix/{filename}",
                    "Body": ANY,
                    "ContentType": content_type,
                    "IfNoneMatch": "*",
                },
            )
        candidate.upload_files(s3, "bucket", "prefix")
        stub.assert_no_pending_responses()


def test_upload_conflict_stops_before_metadata_or_success_summary(actions_env, monkeypatch):
    candidate.ARTIFACT.parent.mkdir()
    candidate.ARTIFACT.write_bytes(b"model")
    candidate.METADATA.write_text("{}")
    s3 = Mock()
    s3.put_object.side_effect = ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
    monkeypatch.setattr(common, "client", lambda _: s3)
    metadata = Mock()
    monkeypatch.setattr(candidate, "write_metadata", metadata)
    monkeypatch.setenv("MODEL_PREFIX", "production/forbidden")
    with pytest.raises(ClientError):
        candidate.upload()
    assert s3.put_object.call_count == 1
    assert s3.put_object.call_args.kwargs["Key"] == "models/views/candidates/123/2/model.joblib"
    assert not Path("summary").exists()
    assert metadata.call_args.kwargs["postings_version"] == "v-p"


def test_copy_existing_digest_does_not_push(monkeypatch):
    docker = Mock()
    monkeypatch.setattr(common, "command", docker)
    ecr = aws("ecr")
    with Stubber(ecr) as stub:
        stub.add_response(
            "batch_get_image",
            {"images": [{"imageId": {"imageDigest": DIGEST}}]},
            {"repositoryName": "views-model", "imageIds": [{"imageDigest": DIGEST}]},
        )
        assert deploy.copy_image(ecr, SOURCE, REGISTRY, "views-model") == IMAGE
    docker.assert_not_called()


@pytest.mark.parametrize("copied", [DIGEST, "sha256:" + "c" * 64])
def test_copy_requires_identical_digest(monkeypatch, copied):
    docker = Mock()
    monkeypatch.setattr(common, "command", docker)
    ecr = aws("ecr")
    with Stubber(ecr) as stub:
        stub.add_response("batch_get_image", {"failures": [{"failureCode": "ImageNotFound"}]})
        stub.add_response("describe_images", {"imageDetails": [{"imageDigest": copied}]})
        if copied == DIGEST:
            assert deploy.copy_image(ecr, SOURCE, REGISTRY, "views-model") == IMAGE
        else:
            with pytest.raises(ValueError, match="digest changed"):
                deploy.copy_image(ecr, SOURCE, REGISTRY, "views-model")
    assert [call.args[1] for call in docker.call_args_list] == ["pull", "tag", "push"]


def test_ecr_access_failure_is_not_treated_as_absent(monkeypatch):
    docker = Mock()
    monkeypatch.setattr(common, "command", docker)
    ecr = Mock()
    ecr.batch_get_image.return_value = {"failures": [{"failureCode": "KmsError"}]}
    with pytest.raises(RuntimeError, match="lookup failed"):
        deploy.copy_image(ecr, SOURCE, REGISTRY, "views-model")
    docker.assert_not_called()


def test_task_revision_preserves_model_config_and_sidecar():
    before = deepcopy(TASK)
    updated = deploy.task_revision(TASK, IMAGE)
    assert TASK == before
    assert "taskDefinitionArn" not in updated
    assert updated["taskRoleArn"] == "keep-role"
    assert (
        updated["containerDefinitions"][0]["environment"]
        == before["containerDefinitions"][0]["environment"]
    )
    assert updated["containerDefinitions"][0]["image"] == IMAGE
    assert updated["containerDefinitions"][1] == before["containerDefinitions"][1]
    for containers in ([], [before["containerDefinitions"][0]] * 2):
        with pytest.raises(ValueError, match="exactly one"):
            deploy.task_revision({**TASK, "containerDefinitions": containers}, IMAGE)


@pytest.mark.parametrize("active_revision", ["new-task", "old-task"])
def test_rollout_checks_new_revision_after_waiter(active_revision):
    ecs = Mock()
    ecs.describe_services.side_effect = [
        {"services": [{"taskDefinition": "old-task"}]},
        {
            "services": [
                {
                    "taskDefinition": active_revision,
                    "desiredCount": 2,
                    "runningCount": 2,
                    "pendingCount": 0,
                    "deployments": [
                        {"taskDefinition": active_revision, "rolloutState": "COMPLETED"}
                    ],
                }
            ]
        },
    ]
    ecs.describe_task_definition.return_value = {"taskDefinition": TASK}
    ecs.register_task_definition.return_value = {
        "taskDefinition": {"taskDefinitionArn": "new-task"}
    }
    if active_revision == "new-task":
        assert deploy.deploy_revision(ecs, "cluster", "service", IMAGE) == ("old-task", "new-task")
    else:
        with pytest.raises(ValueError, match="intended task"):
            deploy.deploy_revision(ecs, "cluster", "service", IMAGE)
    ecs.get_waiter.assert_called_once_with("services_stable")
    ecs.update_service.assert_called_once_with(
        cluster="cluster", service="service", taskDefinition="new-task"
    )


def test_invalid_task_never_registers_or_updates():
    ecs = Mock()
    ecs.describe_services.return_value = {"services": [{"taskDefinition": "old"}]}
    ecs.describe_task_definition.return_value = {
        "taskDefinition": {**TASK, "containerDefinitions": []}
    }
    with pytest.raises(ValueError):
        deploy.deploy_revision(ecs, "cluster", "service", IMAGE)
    ecs.register_task_definition.assert_not_called()
    ecs.update_service.assert_not_called()


@pytest.mark.parametrize("reference", [SOURCE, "ghcr.io/other/repo@" + DIGEST])
def test_publish_saves_only_expected_registry_digest(actions_env, monkeypatch, reference):
    docker = Mock(
        side_effect=["loaded", "tagged", "pushed", json.dumps([{"RepoDigests": [reference]}])]
    )
    monkeypatch.setattr(common, "command", docker)
    if reference == SOURCE:
        release.publish()
        assert json.loads(Path("release.json").read_text()) == RELEASE
        assert SOURCE in Path("summary").read_text()
    else:
        with pytest.raises(ValueError, match="exactly one"):
            release.publish()
        assert not Path("release.json").exists()
    assert [call.args[1] for call in docker.call_args_list] == ["load", "tag", "push", "image"]


def test_publication_failure_does_not_write_manifest(actions_env, monkeypatch):
    monkeypatch.setattr(
        common, "command", Mock(side_effect=subprocess.CalledProcessError(1, ["docker"]))
    )
    with pytest.raises(subprocess.CalledProcessError):
        release.publish()
    assert not Path("release.json").exists()


@pytest.mark.parametrize("conclusion", ["success", "failure"])
def test_ci_evidence_download_writes_only_verified_output(actions_env, monkeypatch, conclusion):
    def gh(*args):
        if args[1] == "api":
            return json.dumps({**RUN, "conclusion": conclusion})
        (Path(args[-1]) / "release.json").write_text(json.dumps(RELEASE))
        return ""

    monkeypatch.setattr(common, "command", gh)
    if conclusion == "success":
        release.verify()
        assert Path("outputs").read_text() == f"image={SOURCE}\n"
    else:
        with pytest.raises(ValueError):
            release.verify()
        assert not Path("outputs").exists()
