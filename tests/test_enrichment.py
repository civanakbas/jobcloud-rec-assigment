import json
import subprocess
import sys
from copy import deepcopy

import pytest

from mlops_assignment.ai_exercise.provider import ScriptedEnrichmentProvider
from mlops_assignment.enrichment.cli import EXERCISE, read_jsonl, run_curated
from mlops_assignment.enrichment.taxonomy import Taxonomy
from mlops_assignment.enrichment.validation import PayloadValidator
from mlops_assignment.enrichment.workflow import EnrichmentWorkflow
from mlops_assignment.reports import evaluate, index_records

SCHEMA = json.loads((EXERCISE / "enrichment.schema.json").read_text())
TAXONOMY = json.loads((EXERCISE / "taxonomy.json").read_text())
SCENARIOS = json.loads((EXERCISE / "provider_scenarios.json").read_text())["scenarios"]
POSTINGS = index_records(
    read_jsonl(EXERCISE / "postings.jsonl") + read_jsonl(EXERCISE / "edge_cases.jsonl")
)
VALID = {
    "occupation_id": "data_engineer",
    "seniority": "senior",
    "skill_ids": ["python"],
    "work_arrangement": "remote",
    "confidence": 0.8,
}


@pytest.fixture
def workflow():
    return EnrichmentWorkflow(SCHEMA, TAXONOMY, "test-provider-v1")


@pytest.mark.parametrize("name", SCENARIOS)
def test_supplied_scenarios(workflow, name):
    scenario = SCENARIOS[name]
    provider = ScriptedEnrichmentProvider.from_scenario(EXERCISE / "provider_scenarios.json", name)
    result = workflow.run(POSTINGS[scenario["posting_id"]], provider)
    assert result["state"] == scenario["expected_state"]
    assert result["attempt_count"] == len(provider.requests) <= 2
    assert result["schema_valid"] == (result["state"] != "fallback")
    if result["attempt_count"] == 2:
        assert provider.requests[1].validation_feedback
        assert provider.requests[1].title == provider.requests[0].title
    if name == "malformed_then_valid":
        assert result["attempt_count"] == scenario["expected_attempt_count"]
        assert result["validation_errors"][0]["attempt"] == 1
    if name == "unknown_taxonomy_concept":
        assert result["unresolved"] == {
            "occupation_id": "growth_hacker",
            "skill_ids": ["growth_magic"],
        }


def test_alias_mapping_dedup_and_provider_values(workflow):
    payload = {
        **VALID,
        "occupation_id": "rN",
        "skill_ids": ["Python 3", "PYTHON", "pYtHoN", "MICU"],
    }
    provider = ScriptedEnrichmentProvider.from_result(payload)
    result = workflow.run(
        POSTINGS["p002"], provider
    )  # Posting is a data engineer, response is a nurse.
    assert result["state"] == "accepted"
    assert result["enrichment"]["occupation_id"] == "registered_nurse"
    assert result["enrichment"]["skill_ids"] == ["intensive_care", "python"]
    assert payload["skill_ids"] == ["Python 3", "PYTHON", "pYtHoN", "MICU"]


@pytest.mark.parametrize(
    "confidence,state", [(0.799, "review"), (0.8, "accepted"), (1.0, "accepted")]
)
def test_threshold(workflow, confidence, state):
    result = workflow.run(
        POSTINGS["p001"],
        ScriptedEnrichmentProvider.from_result({**VALID, "confidence": confidence}),
    )
    assert result["state"] == state


def test_null_occupation_is_review_even_at_high_confidence(workflow):
    result = workflow.run(
        POSTINGS["p001"], ScriptedEnrichmentProvider.from_result({**VALID, "occupation_id": None})
    )
    assert result["schema_valid"]
    assert result["state"] == "review"
    assert result["reasons"] == ["missing_occupation"]


@pytest.mark.parametrize(
    "patch",
    [
        {"extra": True},
        {"confidence": True},
        {"confidence": float("nan")},
        {"confidence": float("inf")},
        {"skill_ids": ["python", "python"]},
        {"skill_ids": [str(i) for i in range(9)]},
        {"seniority": "experienced"},
    ],
)
def test_invalid_schema_is_not_coerced(workflow, patch):
    provider = ScriptedEnrichmentProvider([{"kind": "result", "value": {**VALID, **patch}}] * 2)
    result = workflow.run(POSTINGS["p001"], provider)
    assert result["state"] == "fallback"
    assert result["reasons"] == ["invalid_output_after_retry"]
    assert len(provider.requests) == 2
    assert len(result["validation_errors"]) == 2


def test_nonmapping_payload_validation():
    assert PayloadValidator(SCHEMA).errors([1, 2])


@pytest.mark.parametrize(
    "kind,reason", [("error", "provider_failure"), ("timeout", "provider_timeout")]
)
def test_provider_failure_is_not_retried(workflow, kind, reason):
    provider = ScriptedEnrichmentProvider([{"kind": kind, "message": "private provider details"}])
    result = workflow.run(POSTINGS["p001"], provider)
    assert result["state"] == "fallback"
    assert result["reasons"] == [reason]
    assert result["attempt_count"] == 1
    assert "private provider details" not in json.dumps(result)


def test_failure_on_retry_is_bounded(workflow):
    provider = ScriptedEnrichmentProvider([{"kind": "result", "value": {}}, {"kind": "error"}])
    result = workflow.run(POSTINGS["p001"], provider)
    assert result["state"] == "fallback"
    assert result["attempt_count"] == 2
    assert result["reasons"] == ["provider_failure"]


@pytest.mark.parametrize("posting_id", ["e001", "e002", "e003", "e004"])
def test_constructed_edge_inputs_are_passed_without_extraction(workflow, posting_id):
    posting = dict(POSTINGS[posting_id])
    if "description_template" in posting:
        posting["description"] = posting["description_template"] * posting["repeat"]
    provider = ScriptedEnrichmentProvider.from_result(VALID)
    result = workflow.run(posting, provider)
    assert (
        result["state"] == "accepted"
    )  # State follows the scripted response, not our interpretation of text.
    assert provider.requests[0].description == posting.get("description")
    assert provider.requests[0].title == posting["title"]


def test_taxonomy_ambiguity_is_configuration_error():
    taxonomy = deepcopy(TAXONOMY)
    taxonomy["skills"][0]["aliases"].append("Python")
    with pytest.raises(ValueError, match="Ambiguous"):
        Taxonomy(taxonomy)


def test_version_changes_with_meaningful_inputs(workflow):
    assert workflow.version == EnrichmentWorkflow(SCHEMA, TAXONOMY, "test-provider-v1").version
    assert workflow.version != EnrichmentWorkflow(SCHEMA, TAXONOMY, "different-provider").version
    assert workflow.version != EnrichmentWorkflow(SCHEMA, TAXONOMY, "test-provider-v1", 0.9).version
    taxonomy = deepcopy(TAXONOMY)
    taxonomy["skills"][0]["aliases"].append("new alias")
    assert workflow.version != EnrichmentWorkflow(SCHEMA, taxonomy, "test-provider-v1").version
    schema = deepcopy(SCHEMA)
    schema["properties"]["skill_ids"]["maxItems"] = 7
    assert workflow.version != EnrichmentWorkflow(schema, TAXONOMY, "test-provider-v1").version


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -1, 2])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="confidence_threshold"):
        EnrichmentWorkflow(SCHEMA, TAXONOMY, "test", threshold)


def test_curated_evaluation_reproducible_and_joins_by_posting_id():
    results, manifest = run_curated()
    assert (results, manifest) == run_curated()
    labels = read_jsonl(EXERCISE / "evaluation_labels.jsonl")
    report = evaluate(list(reversed(results)), labels)
    assert report["postings"] == 10
    assert report["schema_validity_rate"] == 1.0
    assert report["occupation_accuracy"] == 1.0
    assert report["skills"]["f1"] == 1.0


def test_evaluation_keeps_review_fallback_and_unknown_skills(workflow):
    review = workflow.run(
        POSTINGS["p001"],
        ScriptedEnrichmentProvider.from_result(
            {**VALID, "confidence": 0.2, "skill_ids": ["python", "unknown"]}
        ),
    )
    fallback = workflow.run(POSTINGS["p002"], ScriptedEnrichmentProvider([{"kind": "timeout"}]))
    labels = [
        {"posting_id": key, "occupation_id": "data_engineer", "skill_ids": ["python"]}
        for key in ("p001", "p002")
    ]
    report = evaluate([review, fallback], labels)
    assert report["schema_validity_rate"] == report["occupation_accuracy"] == 0.5
    assert report["skills"]["precision"] == report["skills"]["recall"] == 0.5
    assert report["states"] == {"accepted": 0, "review": 1, "fallback": 1}
    with pytest.raises(ValueError, match="duplicate"):
        evaluate([review, review], labels)
    with pytest.raises(ValueError, match="identical"):
        evaluate([review], labels)


def test_cli_executes_and_writes_report(tmp_path):
    out, report = tmp_path / "results.jsonl", tmp_path / "report.json"
    command = [
        sys.executable,
        "-m",
        "mlops_assignment.enrichment.cli",
        "--out",
        str(out),
        "--report",
        str(report),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)
    original = out.read_bytes(), report.read_bytes()
    assert len(read_jsonl(out)) == 10
    assert json.loads(report.read_text())["postings"] == 10
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)
    assert original == (out.read_bytes(), report.read_bytes())
