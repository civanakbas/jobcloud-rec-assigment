"""Provider boundary supplied for the AI enrichment exercise."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class EnrichmentRequest:
    title: str
    description: str | None
    validation_feedback: tuple[str, ...] = ()


class EnrichmentProvider(Protocol):
    """Return an untrusted provider payload for validation by the workflow."""

    def enrich(self, request: EnrichmentRequest) -> Mapping[str, Any]: ...


class ProviderError(RuntimeError):
    """A scripted provider failure."""


class ProviderTimeout(TimeoutError):
    """A scripted provider timeout."""


class ScriptedEnrichmentProvider:
    """Replay deterministic attempts without implementing workflow decisions."""

    def __init__(self, attempts: Sequence[Mapping[str, Any]]) -> None:
        if not attempts:
            raise ValueError("at least one scripted attempt is required")
        self._attempts = list(attempts)
        self.requests: list[EnrichmentRequest] = []

    @classmethod
    def from_scenario(
        cls, path: str | Path, scenario_name: str
    ) -> "ScriptedEnrichmentProvider":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            attempts = document["scenarios"][scenario_name]["attempts"]
        except KeyError as exc:
            raise KeyError(f"unknown provider scenario: {scenario_name}") from exc
        return cls(attempts)

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "ScriptedEnrichmentProvider":
        return cls([{"kind": "result", "value": dict(value)}])

    def enrich(self, request: EnrichmentRequest) -> Mapping[str, Any]:
        self.requests.append(request)
        index = len(self.requests) - 1
        if index >= len(self._attempts):
            raise ProviderError("scripted provider has no remaining attempt")

        attempt = self._attempts[index]
        kind = attempt.get("kind")
        if kind == "result":
            value = attempt.get("value")
            if not isinstance(value, Mapping):
                raise ProviderError("scripted result must be a mapping")
            return dict(value)
        if kind == "timeout":
            raise ProviderTimeout(str(attempt.get("message", "provider timed out")))
        if kind == "error":
            raise ProviderError(str(attempt.get("message", "provider failed")))
        raise ProviderError(f"unsupported scripted attempt kind: {kind!r}")
