# AI enrichment exercise data

This directory contains the supplied starter material for the MLOps and AI
Engineering Take-Home PDF. It is packaged under `mlops_assignment.ai_exercise`.
Provider and fixture contents are unchanged.

## Files

- `postings.jsonl`: ten curated posting excerpts from the supplied Kaggle dataset.
- `taxonomy.json`: the allowed occupation and skill concepts, including aliases.
- `evaluation_labels.jsonl`: expected labels for the curated postings.
- `edge_cases.jsonl`: constructed input cases for review and fallback behavior.
- `provider_scenarios.json`: deterministic provider results and failures for tests.
- `evaluation_provider_outputs.jsonl`: deterministic successful outputs for evaluation.
- `enrichment.schema.json`: the required structured-output contract.
- `provider.py`: provider protocol, exceptions, and a scripted provider implementation.

The `source_job_id` values link the selected postings back to `data/postings.csv` in
the full source package. Titles and posting metadata come from those rows. Descriptions
are shortened and may be lightly adapted for readability, so use `source_job_id` when
exact source text matters. Labels were prepared for this exercise; they are not part
of the Kaggle dataset and are not a statistically representative or production-quality
gold standard.

Files under `edge_cases.jsonl` and failure entries in `provider_scenarios.json` are
explicitly constructed. They must not be presented as source-dataset examples or as
evidence of provider quality.

The evaluation provider outputs intentionally align closely with the exercise labels.
They exist to make the evaluation pipeline executable and are not evidence of LLM
quality.

All files are UTF-8. JSONL files contain one JSON object per line.
