"""Copy verified images and deploy ECS task revisions without rebuilding."""

import copy
import re

import fire

from mlops_assignment.delivery import common


def copy_image(ecr, source: str, registry: str, repository: str) -> str:
    if not re.fullmatch(r"ghcr\.io/[a-z0-9_.-]+/[a-z0-9_.-]+@sha256:[a-f0-9]{64}", source):
        raise ValueError("Expected a verified GHCR digest reference")
    if not re.fullmatch(r"[a-z0-9.-]+", registry) or not re.fullmatch(
        r"[a-z0-9][a-z0-9._/-]*", repository
    ):
        raise ValueError("Invalid ECR image location")
    digest = common.digest(source.split("@", 1)[1])
    response = ecr.batch_get_image(repositoryName=repository, imageIds=[{"imageDigest": digest}])
    failures = response.get("failures", [])
    if any(failure.get("failureCode") != "ImageNotFound" for failure in failures):
        raise RuntimeError("ECR lookup failed; refusing to treat it as an absent image")
    images = response.get("images", [])
    if images:
        if len(images) != 1 or images[0]["imageId"]["imageDigest"] != digest or failures:
            raise ValueError("ECR returned unexpected image evidence")
    else:
        tag = "sha256-" + digest.removeprefix("sha256:")
        destination = f"{registry}/{repository}:{tag}"
        common.command("docker", "pull", source)
        common.command("docker", "tag", source, destination)
        common.command("docker", "push", destination)
        details = ecr.describe_images(repositoryName=repository, imageIds=[{"imageTag": tag}])
        if (
            len(details.get("imageDetails", [])) != 1
            or details["imageDetails"][0]["imageDigest"] != digest
        ):
            raise ValueError("Copied image digest changed; refusing deployment")
    return f"{registry}/{repository}@{digest}"


def copy_to_ecr() -> None:
    source = common.required("SOURCE_IMAGE")
    registry = common.required("ECR_REGISTRY")
    repository = common.required("ECR_REPOSITORY")
    common.required("GITHUB_OUTPUT")
    image = copy_image(common.client("ecr"), source, registry, repository)
    common.output("image", image)


def task_revision(task: dict, image: str) -> dict:
    if not re.fullmatch(r"[a-z0-9.-]+/[a-z0-9][a-z0-9._/-]*@sha256:[a-f0-9]{64}", image):
        raise ValueError("Task image must be an immutable digest reference")
    revision = copy.deepcopy(task)
    containers = [
        item for item in revision["containerDefinitions"] if item["name"] == "views-model"
    ]
    if len(containers) != 1:
        raise ValueError("Expected exactly one views-model container")
    containers[0]["image"] = image
    for key in (
        "taskDefinitionArn",
        "revision",
        "status",
        "requiresAttributes",
        "compatibilities",
        "registeredAt",
        "registeredBy",
        "deregisteredAt",
    ):
        revision.pop(key, None)
    return revision


def service_state(response: dict) -> dict:
    if response.get("failures") or len(response.get("services", [])) != 1:
        raise ValueError("Expected exactly one available ECS service")
    return response["services"][0]


def verify_rollout(service: dict, task_arn: str) -> None:
    deployments = service.get("deployments", [])
    if not (
        service.get("taskDefinition") == task_arn
        and service.get("desiredCount", 0) > 0
        and service.get("runningCount") == service["desiredCount"]
        and service.get("pendingCount") == 0
        and len(deployments) == 1
        and deployments[0].get("taskDefinition") == task_arn
        and deployments[0].get("rolloutState") == "COMPLETED"
    ):
        raise ValueError("ECS did not complete the intended task revision")


def deploy_revision(ecs, cluster: str, service: str, image: str) -> tuple[str, str]:
    previous = service_state(ecs.describe_services(cluster=cluster, services=[service]))[
        "taskDefinition"
    ]
    task = ecs.describe_task_definition(taskDefinition=previous)["taskDefinition"]
    revision = task_revision(task, image)
    new = ecs.register_task_definition(**revision)["taskDefinition"]["taskDefinitionArn"]
    ecs.update_service(cluster=cluster, service=service, taskDefinition=new)
    ecs.get_waiter("services_stable").wait(
        cluster=cluster, services=[service], WaiterConfig={"Delay": 15, "MaxAttempts": 80}
    )
    verify_rollout(service_state(ecs.describe_services(cluster=cluster, services=[service])), new)
    return previous, new


def rollout() -> None:
    cluster, service, image = (
        common.required(name) for name in ("ECS_CLUSTER", "ECS_SERVICE", "IMAGE")
    )
    common.required("GITHUB_STEP_SUMMARY")
    previous, new = deploy_revision(common.client("ecs"), cluster, service, image)
    common.summary(f"Previous task: `{previous}`\n\nNew task: `{new}`\n\nImage: `{image}`")


if __name__ == "__main__":
    fire.Fire({"copy": copy_to_ecr, "rollout": rollout})
