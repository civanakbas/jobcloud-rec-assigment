import json

import pytest

from mlops_assignment.ai_exercise.provider import (
    EnrichmentRequest,
    ProviderError,
    ProviderTimeout,
    ScriptedEnrichmentProvider,
)
from mlops_assignment.enrichment.cli import EXERCISE


def _json_lines(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (EXERCISE / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_ai_exercise_fixtures_are_internally_consistent():
    postings = _json_lines("postings.jsonl")
    edge_cases = _json_lines("edge_cases.jsonl")
    labels = _json_lines("evaluation_labels.jsonl")
    evaluation_outputs = _json_lines("evaluation_provider_outputs.jsonl")
    taxonomy = json.loads((EXERCISE / "taxonomy.json").read_text(encoding="utf-8"))
    scenarios = json.loads((EXERCISE / "provider_scenarios.json").read_text(encoding="utf-8"))[
        "scenarios"
    ]
    schema = json.loads((EXERCISE / "enrichment.schema.json").read_text(encoding="utf-8"))

    posting_ids = {item["posting_id"] for item in postings}
    edge_case_ids = {item["posting_id"] for item in edge_cases}
    assert len(posting_ids) == len(postings) == 10
    assert len(edge_case_ids) == len(edge_cases) == 4
    assert posting_ids.isdisjoint(edge_case_ids)

    occupation_ids = {item["id"] for item in taxonomy["occupations"]}
    skill_ids = {item["id"] for item in taxonomy["skills"]}
    assert taxonomy["taxonomy_version"]
    assert schema["additionalProperties"] is False

    assert {item["posting_id"] for item in labels} == posting_ids
    assert {item["posting_id"] for item in evaluation_outputs} == posting_ids
    for label in labels:
        assert label["occupation_id"] in occupation_ids
        assert set(label["skill_ids"]) <= skill_ids

    all_input_ids = posting_ids | edge_case_ids
    assert scenarios
    for scenario in scenarios.values():
        assert scenario["posting_id"] in all_input_ids
        assert scenario["attempts"]
        assert scenario["expected_state"]


def test_scripted_provider_replays_results_and_records_feedback():
    provider = ScriptedEnrichmentProvider.from_scenario(
        EXERCISE / "provider_scenarios.json", "malformed_then_valid"
    )

    first = provider.enrich(EnrichmentRequest("Staff Accountant", "Accounting"))
    second = provider.enrich(
        EnrichmentRequest("Staff Accountant", "Accounting", ("seniority is invalid",))
    )

    assert first["seniority"] == "experienced"
    assert second["seniority"] == "unknown"
    assert provider.requests[1].validation_feedback == ("seniority is invalid",)
    with pytest.raises(ProviderError, match="no remaining attempt"):
        provider.enrich(EnrichmentRequest("Staff Accountant", "Accounting"))


def test_scripted_provider_simulates_timeout():
    provider = ScriptedEnrichmentProvider.from_scenario(
        EXERCISE / "provider_scenarios.json", "timeout"
    )

    with pytest.raises(ProviderTimeout, match="deadline exceeded"):
        provider.enrich(EnrichmentRequest("Software Engineer", "Software"))
