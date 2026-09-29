"""Run the curated fixture through the supplied provider, optionally evaluating it."""

import json
from pathlib import Path

import fire

from mlops_assignment import ai_exercise
from mlops_assignment.ai_exercise.provider import ScriptedEnrichmentProvider
from mlops_assignment.enrichment.workflow import EnrichmentWorkflow, digest
from mlops_assignment.reports import evaluate, index_records

EXERCISE = Path(ai_exercise.__file__).parent


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def run_curated(confidence_threshold: float = 0.8) -> tuple[list[dict], dict]:
    postings = read_jsonl(EXERCISE / "postings.jsonl")
    outputs = index_records(read_jsonl(EXERCISE / "evaluation_provider_outputs.jsonl"))
    if index_records(postings).keys() != outputs.keys():
        raise ValueError(
            "Provider outputs and curated postings must have identical posting_id sets"
        )
    workflow = EnrichmentWorkflow(
        json.loads((EXERCISE / "enrichment.schema.json").read_text(encoding="utf-8")),
        json.loads((EXERCISE / "taxonomy.json").read_text(encoding="utf-8")),
        provider_version="supplied-scripted:"
        + digest(
            {
                "implementation": (EXERCISE / "provider.py").read_text(encoding="utf-8"),
                "outputs": outputs,
            }
        ),
        confidence_threshold=confidence_threshold,
    )
    results = [
        workflow.run(
            posting, ScriptedEnrichmentProvider.from_result(outputs[posting["posting_id"]]["value"])
        )
        for posting in postings
    ]
    manifest = {
        "version": workflow.version,
        "version_inputs": workflow.version_inputs,
        "postings_sha256": digest(postings),
    }
    return results, manifest


def enrich(out: str, report: str | None = None, confidence_threshold: float = 0.8) -> None:
    """Enrich curated postings; write JSONL results and an optional evaluation report."""
    out_path = Path(out)
    report_path = Path(report) if report is not None else None
    protected = {path.resolve() for path in EXERCISE.iterdir() if path.is_file()}
    if out_path.resolve() in protected or (report_path and report_path.resolve() in protected):
        raise ValueError("Output files must not overwrite supplied fixtures")
    if report_path and out_path.resolve() == report_path.resolve():
        raise ValueError("Results and report paths must differ")
    results, manifest = run_curated(confidence_threshold)
    evaluation_report = None
    if report_path:
        labels = read_jsonl(EXERCISE / "evaluation_labels.jsonl")
        evaluation_report = {
            **evaluate(results, labels),
            **manifest,
            "labels_sha256": digest(labels),
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in results
        ),
        encoding="utf-8",
    )
    if evaluation_report is not None and report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                evaluation_report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "postings": len(results),
                "version": manifest["version"],
                "out": str(out_path),
                "report": str(report_path) if report_path else None,
            }
        )
    )


def main() -> None:
    fire.Fire(enrich)


if __name__ == "__main__":
    main()
