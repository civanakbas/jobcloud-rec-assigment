import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

from mlops_assignment.ai_exercise.provider import (
    EnrichmentProvider,
    EnrichmentRequest,
    ProviderError,
    ProviderTimeout,
)
from mlops_assignment.enrichment.taxonomy import Taxonomy
from mlops_assignment.enrichment.validation import PayloadValidator


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


class EnrichmentWorkflow:
    def __init__(
        self, schema: dict, taxonomy: dict, provider_version: str, confidence_threshold: float = 0.8
    ):
        if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be finite and between zero and one")
        self.threshold = confidence_threshold
        self.validator = PayloadValidator(schema)
        self.taxonomy = Taxonomy(taxonomy)
        directory = Path(__file__).parent
        self.version_inputs = {
            "schema_sha256": digest(schema),
            "taxonomy_sha256": digest(taxonomy),
            "taxonomy_version": taxonomy["taxonomy_version"],
            "workflow": "enrichment-v1",
            "confidence_threshold": confidence_threshold,
            "max_attempts": 2,
            "provider_version": provider_version,
            "prompt": "title-and-description-verbatim-v1",
            "implementation_sha256": digest(
                {
                    name: (directory / name).read_text(encoding="utf-8")
                    for name in ("workflow.py", "validation.py", "taxonomy.py")
                }
            ),
        }
        self.version = "enrichment-v1:" + digest(self.version_inputs)

    def run(self, posting: dict, provider: EnrichmentProvider) -> dict:
        request = EnrichmentRequest(posting["title"], posting.get("description"))
        result = {
            "posting_id": posting["posting_id"],
            "source_job_id": posting.get("source_job_id"),
            "version": self.version,
            "state": "fallback",
            "schema_valid": False,
            "attempt_count": 0,
            "enrichment": None,
            "reasons": [],
            "unresolved": {"occupation_id": None, "skill_ids": []},
            "validation_errors": [],
        }
        for attempt in range(1, 3):
            result["attempt_count"] = attempt
            try:
                payload = provider.enrich(request)
            except ProviderTimeout:
                result["reasons"] = ["provider_timeout"]
                return result
            except ProviderError:
                result["reasons"] = ["provider_failure"]
                return result
            errors = self.validator.errors(payload)
            if errors:
                result["validation_errors"].append({"attempt": attempt, "errors": list(errors)})
                request = replace(request, validation_feedback=errors)
                continue
            normalized, unresolved = self.taxonomy.normalize(payload)
            result.update(schema_valid=True, enrichment=normalized, unresolved=unresolved)
            if normalized["occupation_id"] is None:
                result["reasons"].append(
                    "missing_occupation"
                    if payload["occupation_id"] is None
                    else "unresolved_occupation"
                )
            if unresolved["skill_ids"]:
                result["reasons"].append("unresolved_skills")
            if payload["confidence"] < self.threshold:
                result["reasons"].append("low_confidence")
            result["state"] = "review" if result["reasons"] else "accepted"
            return result
        result["reasons"] = ["invalid_output_after_retry"]
        return result
