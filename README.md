# MLOps and AI engineering assignment

An installed Python package serving the inherited job-posting views model, plus a
bounded enrichment workflow using the supplied scripted provider. No live LLM or
full dataset is required for verification.

| Status | Scope |
|---|---|
| Verified locally | Packaging, model loading, Docker/Compose HTTP serving, enrichment/evaluation, 102 tests, Ruff/ty and workflow static validation |
| Implemented; remote integration | CI publication, manual ECS deployment and S3 candidate retraining |
| Design only | Production model promotion, live enrichment, workers/cache, enrichment features and cloud infrastructure |

## Run and verify

Requires **Python 3.12, uv, Make**, and **Docker with Compose** for container checks.
Run from the repository root. Installation/image preparation may need network;
tests and the container smoke need no external services or credentials afterward.
Use a current Compose release supporting [override resets](https://docs.docker.com/reference/compose-file/merge/#reset-value) (verified with 5.5.1).

```bash
make sync
make check test

# Self-contained container verification; leaves the supplied model untouched.
uv run python -m tests.support --out artifacts/synthetic.joblib
API_IMAGE=mlops-assignment:ci docker compose build api
uv run python scripts/smoke_container.py \
  --image mlops-assignment:ci --artifact artifacts/synthetic.joblib

uv run views-enrich \
  --out reports/enrichment.jsonl --report reports/enrichment-evaluation.json
```

For interactive serving, `make up` uses `artifacts/views_baseline.joblib` from the
original archive; it is Git-ignored. If absent, generate a synthetic replacement
with `uv run python -m tests.support --out artifacts/views_baseline.joblib`.
Open `http://localhost:8000/docs`; use `make logs` and `make down` for lifecycle.
Compose binds to loopback and mounts the model read-only. `API_PORT=8001 make up`
changes the host port. `MODEL_ARTIFACT` and `API_IMAGE` select the model and image.
The smoke script uses a unique Compose project and `compose.smoke.yaml` to disable
networking; `compose run` publishes no ports. It forbids builds/pulls, runs HTTP
checks inside the container, then removes the project, including on failure.

`make check` runs Ruff lint, format checking and ty;
**lint applies Ruff fixes**. `make format` formats code. Dependencies are locked in
`uv.lock`; Make checks/tests use `uv run --locked --offline`. Fire exposes the CLIs
(`uv run views-train -- --help`). Local development uses `.venv`; Docker installs
the local project with `uv sync --locked --no-dev --no-editable` into **system Python**, with no venv or repository source tree, as UID 10001.

## Serving

[Prediction](mlops_assignment/predict.py) reuses the fitted preprocessing and
metadata in the model bundle. Imports use the installed namespace, e.g.
`from mlops_assignment.reports import evaluate`.

| Endpoint | Contract |
|---|---|
| `POST /predict` | Batch inference; `views` and `views_per_day` both return total views |
| `GET /health`, `GET /ready` | Liveness and loaded-model/schema readiness |
| `GET /model-info`, `GET /metrics` | Stored metadata/metrics and Prometheus request counts/latencies |

Startup fails on missing/incompatible artifacts. Empty batches return 400; invalid
request schemas/targets return 422. Metric labels use bounded route/method values.
Set `VIEWS_MODEL_ARTIFACT` for a local model, or both `VIEWS_MODEL_S3_BUCKET` and
`VIEWS_MODEL_S3_KEY` for download when absent (`AWS_REGION` selects the region).
Only load trusted joblib artifacts: post-load validation does not secure deserialization.

## LLM enrichment workflow

Implements enrichment workflow using only the ten supplied curated postings and deterministic
provider outputs. It produces occupation, seniority, skills, work arrangement and
confidence, without credentials, network access or a live LLM. The PDF's
`ai_exercise/` fixtures are packaged under `mlops_assignment/ai_exercise/`.

| Implementation | Responsibility |
|---|---|
| [cli.py](mlops_assignment/enrichment/cli.py) | `views-enrich` → `run_curated()`: load postings and configure each scripted provider |
| [provider.py](mlops_assignment/ai_exercise/provider.py) | Supplied `ScriptedEnrichmentProvider`; workflow accepts the `EnrichmentProvider` interface |
| [workflow.py](mlops_assignment/enrichment/workflow.py) | `EnrichmentWorkflow.run()`: provider calls, retry feedback, state and reason recording |
| [validation.py](mlops_assignment/enrichment/validation.py) | Validate against the supplied `enrichment.schema.json` before normalization |
| [taxonomy.py](mlops_assignment/enrichment/taxonomy.py) | Map provider values against the closed vocabulary in `taxonomy.json` |
| [reports.py](mlops_assignment/reports.py) | Join labels and calculate schema validity and field-level metrics |

```text
curated posting -> provider.enrich(request) -> schema validation -> taxonomy -> state
                                                  |
                                      invalid: one retry with feedback
```

IDs and explicit aliases match case-insensitively; mapped skills are deduplicated,
and unmatched raw values are preserved under `unresolved`. Mapping uses provider
output, not words extracted directly from posting text. The small fixture taxonomy
is not intended to cover the full dataset.

| Final state | Policy |
|---|---|
| `accepted` | Schema-valid, fully resolved, confidence >= 0.80 |
| `review` | Valid but low confidence, null occupation or unresolved concepts |
| `fallback` | Provider failure/timeout immediately, or invalid output after two attempts |

Only schema-invalid output is retried, at most once, with validation feedback in
the next request. Results retain final reasons, attempt counts and validation
history; fallback invents no enrichment. The confidence threshold is configurable
with `--confidence-threshold` and is a policy, not a calibrated probability.
The version fingerprints schema, taxonomy, workflow/normalization/validation code,
provider implementation/outputs, request contract, threshold and retry policy.

Run the batch with the command in **Run and verify**; inspect
[JSONL results](reports/enrichment.jsonl) and the
[evaluation report](reports/enrichment-evaluation.json). For evaluation,
`evaluation_provider_outputs.jsonl` supplies each provider's `value` through
`ScriptedEnrichmentProvider.from_result()`. Each posting follows the same workflow
as scenario tests; labels are used only for scoring.

- Join predictions and labels by `posting_id`; `source_job_id` is traceability only.
- Schema validity counts valid output within allowed attempts over **all postings**.
- Occupation accuracy includes reviews; fallbacks count as incorrect.
- Skills are sets with **micro-averaging**. Reviews are scored, fallbacks predict
  no skills, and unresolved skills count as false positives.

**Recorded:** schema validity 10/10 (100%), occupation accuracy 10/10 (100%),
skill precision/recall/F1 100% (TP/FP/FN = 24/0/0); accepted/review/fallback = 10/0/0.
These deliberately aligned fixtures verify pipeline execution, not provider quality
or production accuracy.

[Deterministic tests](tests/test_enrichment.py) cover alias normalization, correction
on retry, low-confidence/unresolved review, provider failure/timeout, exhausted
retry, robustness cases, version changes and repeatability. Run them independently:

```bash
uv run python -m pytest tests/test_enrichment.py -q
```

Standard Python fits this fixed, bounded sequence and keeps components independently
testable. Timeouts are simulated; a live adapter would need enforced deadlines.
Enrichment is separate from `/predict`: live providers, workers/cache and model
feature integration remain design-only.

## Delivery


[CI](.github/workflows/ci.yml) saves and publishes the tested image without rebuilding.
PRs cannot publish; only publication gets package-write permission. `release.json`
records commit, repository, run/attempt and immutable image digest. Image-transfer
artifacts expire after two days, manifests after 90; rollback needs retained images
and evidence. Tags alone are not immutable identities.

[Deployment](.github/workflows/deploy.yml) takes `ci_run_id` and `image_digest` on main.
It verifies successful matching CI evidence before acquiring AWS credentials,
requires the same digest in ECR, and changes only the existing `views-model`
container image. It checks the intended revision after ECS stability, detecting
rollback to an old revision. Application rollback selects a prior verified release;
model configuration is preserved.

[Retraining](.github/workflows/retrain.yml) takes non-null `postings_version` and
`companies_version` S3 VersionIds. It writes candidates only under
`models/views/candidates/<run-id>/<attempt>/`, with conditional no-overwrite uploads.
Metadata records data versions/hashes, code SHA, lockfile/model hashes and metrics.
Uploads are not transactional: require both model and metadata. There is no automatic
promotion or service restart; stored random-holdout MAEs are not a promotion gate.

Execution lives in [installed Fire modules](mlops_assignment/delivery):
`release publish/verify`, `deploy copy/rollout`, and
`candidate validate/download/train/upload`. YAML retains permissions, credentials
and sequencing; Python uses boto3 and checked subprocess argument lists.

Cloud examples require existing linux/amd64-compatible ECR/ECS resources and GitHub
variables `AWS_REGION`, `ECR_REPOSITORY`, `ECS_CLUSTER`, `ECS_SERVICE`,
`AWS_DEPLOY_ROLE_ARN`, `AWS_TRAIN_ROLE_ARN`, `DATA_BUCKET`, `MODEL_BUCKET`.
Configure environment-scoped OIDC, branch/environment protections and scoped IAM:
deploy needs ECR/ECS and approved `iam:PassRole`; training needs versioned reads and
candidate-prefix writes, without production writes. No infrastructure is provisioned.

## Verification evidence

Latest recorded checks: **2026-09-29**, Python 3.12.14 / uv 0.12.19.
These are historical execution results, not claims of a new run for this documentation edit.

| Executed check | Result |
|---|---|
| `make check test` | Ruff/ty passed; 102 tests in 9.77s; one upstream Starlette/AnyIO warning |
| Wheel build and installed imports | Passed; delivery modules included |
| Compose build and smoke | Passed after moving the package to the repository root; installed-package/system Python assertions, real HTTP, both targets, invalid requests and cleanup |
| `docker compose -p mlops-compose-check up --build --wait --wait-timeout 90` | Earlier healthy startup; host readiness/prediction checks passed; removed with `down` |
| Enrichment command above | Ten results; evaluation counts above; repeatability covered by tests |
| `actionlint` (v1.7.7), `git diff --check` | Passed |

The supplied artifact produced approximately **7.1082 / 8.9191** total views for the
smoke posting's two targets; the synthetic artifact produced **63.8831 / 63.8831**.
These are execution checks, not quality comparisons. The suite used offline
resolution outside the tooling sandbox because its HTTP event-loop thread stalled
inside it. AWS/registry interactions are stubbed. CI was pushed, but its remote
outcome has not been reviewed here; no cloud deployment, load test or full-data
retraining is claimed.

## Production design and next priorities

1. **Verify delivery remotely.** Exercise publication, IAM/OIDC, digest preservation,
   readiness and rollback. Retain release evidence; configure service ownership and
   workflow-failure notifications. Suggested serving: multiple ECS tasks behind an
   ALB, readiness routing, alarms for errors, latency, capacity and artifact age.
2. **Define model semantics and promotion.** The inherited `snapshot_ms` freezes
   posting age. Define `prediction_at` and label-observation horizon; score candidate
   and incumbent on the same versioned temporal population and metric implementation,
   checking leakage and segment regressions. Do not silently substitute wall-clock
   age. Promote trusted immutable model references separately from code; retain
   compatible prior artifacts for rollback.
3. **Add production controls.** Authentication, input limits, artifact provenance
   verification, measured resource/latency limits and alert delivery remain open.
   Base images/build dependencies are not fully digest-pinned. Large models need
   verified downloads, warmup/readiness and capacity for overlapping replicas.
4. **Add live enrichment only if justified.** Use an asynchronous worker and versioned
   cache; first predictions must support missing/delayed enrichment. Historical
   features must reflect availability at prediction time. Version prompts/providers,
   models, schemas, taxonomies, workflows and features together; evaluate against
   independent human labels before rollout. Monitor quality, coverage, queue age,
   deadlines, tokens and cost. Avoid raw text/exception bodies in telemetry; restrict
   unresolved values and retention. Enrichment rollback selects a prior compatible
   version/cache namespace, not arbitrary model/version combinations.




