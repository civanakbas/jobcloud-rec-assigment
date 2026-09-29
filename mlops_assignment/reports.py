"""All-posting evaluation; review/fallback records are never silently excluded."""

from collections import Counter


def index_records(records: list[dict]) -> dict[str, dict]:
    indexed = {}
    for record in records:
        key = record["posting_id"]
        if not isinstance(key, str) or not key or key in indexed:
            raise ValueError(f"Invalid or duplicate posting_id: {key!r}")
        indexed[key] = record
    return indexed


def evaluate(results: list[dict], labels: list[dict]) -> dict:
    predictions, gold = index_records(results), index_records(labels)
    if not gold or predictions.keys() != gold.keys():
        raise ValueError("Evaluation requires nonempty, identical posting_id sets")
    correct = valid = tp = fp = fn = 0
    states = Counter()
    for key, label in gold.items():
        result = predictions[key]
        states[result["state"]] += 1
        valid += int(result["schema_valid"])
        output = result["enrichment"] or {}
        correct += int(
            result["state"] != "fallback" and output.get("occupation_id") == label["occupation_id"]
        )
        # Tag unknown raw concepts so they cannot collide with canonical IDs.
        predicted = {("id", value) for value in output.get("skill_ids", [])}
        predicted |= {
            ("unresolved", value.strip().casefold()) for value in result["unresolved"]["skill_ids"]
        }
        expected = {("id", value) for value in label["skill_ids"]}
        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "postings": len(gold),
        "schema_valid_postings": valid,
        "schema_validity_rate": valid / len(gold),
        "occupation_accuracy": correct / len(gold),
        "skills": {
            "averaging": "micro",
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        },
        "states": {state: states[state] for state in ("accepted", "review", "fallback")},
        "policy": "All postings included; review scored normally; fallback has no predictions; unresolved skills count as false positives; undefined skill ratios are zero.",
        "limitation": "Scripted outputs aligned with exercise labels verify pipeline execution, not LLM quality or production performance.",
    }
