# TAPay

**Intent-grounded trustworthy agentic payment.** TAPay is an engineering prototype for offline payment simulation. It connects three control agents to transaction generation, verification, execution decisions and audit trails:

| Agent | Abbreviation | Responsibility | Module |
|---|---|---|---|
| Structured Intent Representation Agent | SIRA | Versioned intent fields, source evidence, clarification and user revisions | `model/payment/sira/` |
| Transaction-Intent Alignment Agent | TIAA | Independent transaction–intent checks using rules and current evidence | `model/payment/tiaa/` |
| Dynamic Boundary Decision Agent | DBDA | Stateful decision boundaries, event-driven rechecks and current-receipt checks | `model/payment/dbda/` |

The public module directories use the lowercase abbreviations `sira`, `tiaa`, and `dbda`. Historical ablation identifiers H1, H2, and H3 correspond to SIRA, TIAA, and DBDA, respectively. Existing CLI flags (`--h1`, `--h2`, `--h3`) and recorded field/class names are retained for compatibility.

The supported business decisions are `EXECUTE`, `BLOCK`, and `REQUEST_CLARIFICATION`; infrastructure failures are recorded separately. Execution is simulated: this release does not connect to payment rails or move funds.

## Layout

```text
TAPay/
├── datasets/                 # Simulation datasets and metadata
├── model/                    # Core implementation and upstream license
├── config/                   # Runtime settings and credential-free example
├── main.py                   # Entry point
├── requirements.txt
└── README.md
```

## Install

Python 3.11 is recommended. From this directory:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Offline smoke run (no API key)

```bash
python main.py --input datasets/hard_55.jsonl --output outputs/smoke --split all --limit 3 --mock
```

`--mock` uses scripted responses, including reference labels. It tests plumbing and control flow; its scores are **not** empirical LLM results. Output directories must be fresh unless `--resume` is supplied; resume requires the same frozen input, code and settings.

## Run with a model provider

Copy `config/llm.example.ini` to `config/llm.ini`, set `deployment_name` to your available model and optionally set your compatible `base_url`. Export `OPENAI_API_KEY` locally. Never commit keys. Tool calling and structured output support are required.

```bash
python main.py --input datasets/hard_55.jsonl --output outputs/hard-full --split all
python main.py --input datasets/holdout_48.jsonl --output outputs/holdout-full --split all
python main.py --variant backbone --input datasets/hard_55.jsonl --output outputs/hard-backbone --split all
python main.py --variant without-alignment --input datasets/hard_55.jsonl --output outputs/hard-without-alignment --split all
```

Variants: `full` (default), `backbone`, `without-intent`, `without-alignment`, `without-boundary`. The lower-level `--h1`, `--h2`, `--h3` on/off flags remain available for single-agent combinations. `python main.py --help` lists runner options. Real runs use provider APIs and incur provider charges.

Runs produce per-case predictions and trajectories, aggregate metrics, and frozen run identities with source/config hashes. Different model providers, versions and budgets can change outcomes. This minimal release does not bundle historical model outputs or claim bitwise reproduction of reported paper metrics.

## Data

The runtime-ready diagnostic inputs retain mixed review status: 35/55 and 32/48 cases are marked formally eligible. Do not describe these files as wholly human-confirmed; `--require-reviewed` intentionally rejects the full mixed-status inputs.

See [datasets/README.md](datasets/README.md) for provenance, splits, visibility boundaries and limitations. The 311/55/48 collections are not three mutually independent samples. In particular, use `--split all` for all 55 diagnostic cases.

## Evaluation

The decision labels are `EXECUTE`, `BLOCK`, and `REQUEST_CLARIFICATION`. For each dataset, model backbone, method/configuration, and round, let `A` be all assigned case runs and `V` the normally completed runs: those reaching a scorable terminal outcome within the registered budget, with no recorded run failure. Normal completion does not imply a correct or successful decision.

Let `T ⊆ A` be the assignments with an applicable task reference, and `W ⊆ A` those with an applicable workflow reference. Reference eligibility and permitted workflows are fixed before evaluation, independently of model outcomes. The comparison view below uses normal completions; all-assigned results are a separate view. Higher is better (↑), except for FER (↓).

| Metric | Definition and statistical scope |
|---|---|
| Accuracy ↑ | Correct final decision labels in `V`, divided by `n(V)`. This checks the three-way decision, without requiring correct transaction parameters or receipts. |
| Precision ↑ | Macro-average of per-class `TP / (TP + FP)`, using the confusion matrix on `V`. |
| Recall ↑ | Macro-average of per-class `TP / (TP + FN)`, using the confusion matrix on `V`. |
| F1 ↑ | Macro-average of per-class `2TP / (2TP + FP + FN)` on `V`. Calculate each class's F1 before averaging. |
| TSR ↑ | Task success rate over `V ∩ T`, using the decision-specific conditions below. Its denominator is the number of normally completed runs with an applicable task reference, including eligible execution, blocking, and clarification tasks. |
| FER ↓ | Runs in `V` with a reference label of `BLOCK` or `REQUEST_CLARIFICATION` but a prediction of `EXECUTE`, divided by all reference non-execution runs in `V`. In cross-backbone comparisons, compute this separately for each backbone and method. |
| NCR ↑ | Normal completion rate: `n(V) / n(A)`. The denominator includes every assigned case run, including failures; HTTP responses or internal retry attempts are not separate case runs. |
| HF1 ↑ | Per-case F1 between observed and reference directed handoff-edge sets, with repeated edges removed; average the case scores over `V ∩ W`. |
| ASR ↑ | Per-case F1 between observed and reference multisets of directed adjacent-step transitions, retaining repeated-transition counts; average the case scores over `V ∩ W`. |

Precision, recall, and F1 give the three classes equal weight. A zero per-class denominator contributes zero; macro-F1 is not the harmonic mean of macro-precision and macro-recall. For comparisons, display an empty eligible evaluation set as `N/A`, with its effective denominator, rather than interpret it as measured success or failure.

### TSR by reference decision

An applicable task reference must provide the expected decision and the required action contract. Runs without the required reference are marked `REFERENCE_MISSING` and excluded from `T`; they are not automatically successes or failures. For an eligible task, a run failure or an incorrect final decision fails TSR.

| Reference decision | Additional success conditions |
|---|---|
| `EXECUTE` | The selected action and operation are allowed; contract fields marked `PRESENT_VALID` are present and match under the scorer's field-specific comparison rules. At least one stored `SIMULATED_EXECUTED` receipt matches the selected action, applicable parameters, and final environment-state version. The case's simulation-call limit, when specified, is respected. |
| `BLOCK` | The final decision is correct, and its `receipt_id` does not resolve to a successful `SIMULATED_EXECUTED` receipt. |
| `REQUEST_CLARIFICATION` | The final decision is correct, and its `receipt_id` does not resolve to a successful `SIMULATED_EXECUTED` receipt. |

For blocking or clarification, earlier successful simulation probes that the final decision does not cite do not, by themselves, cause TSR to fail. The check concerns the structured final `receipt_id`, not a prohibition on mentioning a simulation in explanatory text. For execution, TSR requires a matching stored receipt. See [`tsr_case`](model/scoring/rq2_metrics.py#L194-L325).

### Workflow references and matching

HF1 preserves edge direction but ignores duplicate handoffs and their global order. For reference and observed edge sets `E` and `O`, precision is `n(E ∩ O) / n(O)` and recall is `n(E ∩ O) / n(E)`; HF1 is their harmonic mean.

ASR measures agreement between observed and reference sequences of workflow roles by counting each directed transition between adjacent steps, including repetitions. The matched count is `m = sum_t min(observed_count(t), reference_count(t))`. Precision divides `m` by the total observed transitions, recall divides it by the total reference transitions, and ASR is their harmonic mean.

When multiple references are applicable, HF1 takes its best match among the fixed reference edge set and legal-trajectory edge sets; ASR takes its best match among the fixed legal trajectories. Each metric selects its own best match. Compute scores per case, then average eligible normal runs; do not pool edges or transitions across cases before calculating F1. Missing workflow references are excluded and their counts reported. See [`hf1_from_edges`, `asr_from_trajectories`, and `workflow_case`](model/scoring/rq2_metrics.py#L352-L440).

### Failures, repeated rounds, and exported summaries

Report the assigned, normal, failed, and reference-eligible counts with each comparison. In the separate all-assigned view, Accuracy uses `A` and TSR uses the fixed reference-eligible assignments `T`; failed runs contribute no success. For all-assigned classification, retain each failed run's reference label and treat its prediction as missing, never as `BLOCK`. For FER, report a separate count of failed runs whose reference decision is `BLOCK` or `REQUEST_CLARIFICATION`.

Calculate every metric within each round first, then use an equal-weight mean across rounds. When reporting dispersion, use the sample standard deviation (`ddof=1`); a single round has no estimated standard deviation. Report the effective number of rounds when a metric is unavailable in some rounds. Report results separately for each dataset, backbone, method, and reference version. State the denominator for each metric in the normal-completion and all-assigned views.

The minimal runner's [`report.py`](model/scoring/report.py) aggregates the records supplied to it: unfiltered output includes failed runs in classification/TSR and any available workflow scores. Use the per-case records to apply the comparison filters and fixed reference sets above; raw aggregate fields are not automatically the normal-completion comparison view. Its case-bootstrap confidence intervals are also distinct from the across-round sample standard deviation.

## License and release scope

Third-party attribution, modification notices and license terms are available in the [license file](model/LICENSE-TalkHier.txt).

This is a minimal source-and-data release, not the full experiment workspace. It excludes literature-baseline implementations, historical runs, raw trajectories, local credentials, provider endpoints, review work files and paper drafts. It therefore does not independently reproduce every comparison table in the manuscript. The mock path uses scripted, label-aware responses for software validation only, never for reporting LLM performance.

No additional license is granted here for original TAPay additions or datasets; their reuse licensing is pending an explicit decision by the rights holder.

Packaging changes relocate `src` to `model`, update code hashing to that path, add a public command wrapper, and translate code comments/docstrings into English.
