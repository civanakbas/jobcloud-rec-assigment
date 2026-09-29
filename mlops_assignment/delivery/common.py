"""Small boundaries for configuration, SDK clients, subprocesses and Actions files."""

import os
import re
import subprocess
from pathlib import Path

import boto3
from botocore.config import Config


def required(name: str) -> str:
    value = os.environ.get(name, "")
    if not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"Missing or invalid environment variable: {name}")
    return value


def positive_id(name: str) -> int:
    value = required(name)
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def repository() -> str:
    value = required("GITHUB_REPOSITORY")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        raise ValueError("Invalid GitHub repository")
    return value


def digest(value: str) -> str:
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", value):
        raise ValueError("Expected a full sha256 image digest")
    return value


def command(*args: str, timeout: int = 600) -> str:
    # Argument arrays avoid shell interpretation; nonzero exits/timeouts propagate.
    result = subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE, timeout=timeout)
    return result.stdout.strip()


def client(service: str):
    return boto3.client(
        service,
        region_name=required("AWS_REGION"),
        config=Config(
            connect_timeout=10,
            read_timeout=120,
            retries={"mode": "standard", "max_attempts": 3},
        ),
    )


def output(name: str, value: str) -> None:
    if "\n" in value or "\r" in value:
        raise ValueError("Actions output must be a single line")
    with Path(required("GITHUB_OUTPUT")).open("a", encoding="utf-8") as stream:
        stream.write(f"{name}={value}\n")


def summary(text: str) -> None:
    with Path(required("GITHUB_STEP_SUMMARY")).open("a", encoding="utf-8") as stream:
        stream.write(text + "\n")
