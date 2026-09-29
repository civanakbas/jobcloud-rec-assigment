"""Lifecycle failures must still remove the isolated Compose project."""

import subprocess
from unittest.mock import Mock

import pytest

from scripts import smoke_container


@pytest.mark.parametrize("failure_stage", [None, "startup", "http"])
def test_compose_smoke_lifecycle(tmp_path, monkeypatch, failure_stage):
    artifact = tmp_path / "model with spaces.joblib"
    artifact.write_bytes(b"fixture")
    calls = []

    def execute(args, **kwargs):
        calls.append((args, kwargs))
        if ("run" in args and failure_stage == "startup") or (
            "exec" in args and failure_stage == "http"
        ):
            raise subprocess.CalledProcessError(1, args)
        return Mock(returncode=0)

    monkeypatch.setattr(smoke_container.subprocess, "run", execute)
    if failure_stage:
        with pytest.raises(subprocess.CalledProcessError):
            smoke_container.smoke("tested:image", str(artifact))
    else:
        smoke_container.smoke("tested:image", str(artifact))
    start, options = calls[0]
    assert start[:2] == ["docker", "compose"]
    assert "--build" not in start and start[start.index("--pull") + 1] == "never"
    assert "--service-ports" not in start
    assert any(arg.endswith("compose.smoke.yaml") for arg in start)
    assert options["env"]["API_IMAGE"] == "tested:image"
    assert options["env"]["MODEL_ARTIFACT"] == str(artifact)
    cleanup, cleanup_options = calls[-1]
    assert "down" in cleanup and "--remove-orphans" in cleanup
    assert cleanup[:8] == start[:8]
    assert cleanup_options["env"] == options["env"]
    if failure_stage:
        assert calls[-2][0][:2] == ["docker", "logs"]
    else:
        assert calls[1][0][:3] == ["docker", "exec", "-i"]
        assert calls[1][1]["input"] == smoke_container.HTTP_CHECK
