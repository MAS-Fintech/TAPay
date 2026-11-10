# TAPay

**Intent-grounded trustworthy agentic payment.** TAPay is an engineering prototype for offline payment simulation. It connects three control agents to transaction generation, verification, execution decisions and audit trails:

| Agent | Responsibility | Internal module |
|---|---|---|
| Structured Intent Agent / 结构化意图Agent | Versioned fields, evidence, clarification and user revisions | `model/payment/h1/` |
| Transaction–Intent Alignment Agent / 交易-意图对齐Agent | Independent rule and evidence-constrained semantic checks | `model/payment/h2/` |
| Dynamic Decision Boundary Agent / 动态决策边界Agent | Stateful risk boundaries and current-receipt checks | `model/payment/h3/` |

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
| TSR ↑ | Ordinary task success rate over `V ∩ T`, using the decision-specific conditions below. Its denominator is the number of normally completed runs with an applicable task reference, including eligible execution, blocking, and clarification tasks. |
| FER ↓ | Runs in `V` with a reference label of `BLOCK` or `REQUEST_CLARIFICATION` but a prediction of `EXECUTE`, divided by all reference non-execution runs in `V`. In cross-backbone comparisons, compute this separately for each backbone and method. |
| NCR ↑ | Normal completion rate: `n(V) / n(A)`. The denominator includes every assigned case run, including failures; HTTP responses or internal retry attempts are not separate case runs. |
| HF1 ↑ | Per-case F1 between observed and reference directed handoff-edge sets, with repeated edges removed; average the case scores over `V ∩ W`. |
| ASR ↑ | Per-case F1 between observed and reference multisets of directed adjacent-step transitions, retaining repeated-transition counts; average the case scores over `V ∩ W`. |

Precision, recall, and F1 give the three classes equal weight. A zero per-class denominator contributes zero; macro-F1 is not the harmonic mean of macro-precision and macro-recall. For comparisons, display an empty eligible evaluation set as `N/A`, with its effective denominator, rather than interpret it as measured success or failure.

### Ordinary TSR: success by reference decision

An applicable task reference must provide the expected decision and the required action contract. Runs without the required reference are marked `REFERENCE_MISSING` and excluded from `T`; they are not automatically successes or failures. For an eligible task, a run failure or an incorrect final decision fails ordinary TSR.

| Reference decision | Additional success conditions |
|---|---|
| `EXECUTE` | The selected action and operation are allowed; contract fields marked `PRESENT_VALID` are present and match under the scorer's field-specific comparison rules. At least one stored `SIMULATED_EXECUTED` receipt matches the selected action, applicable parameters, and final environment-state version. The case's simulation-call limit, when specified, is respected. |
| `BLOCK` | The final decision is correct, and its `receipt_id` does not resolve to a successful `SIMULATED_EXECUTED` receipt. |
| `REQUEST_CLARIFICATION` | The final decision is correct, and its `receipt_id` does not resolve to a successful `SIMULATED_EXECUTED` receipt. |

For blocking or clarification, earlier successful simulation probes that the final decision does not cite do not, by themselves, cause ordinary TSR to fail. The check concerns the structured final `receipt_id`, not a prohibition on mentioning a simulation in explanatory text. For execution, ordinary TSR requires a matching stored receipt; strict TSR additionally checks explicit receipt citation and termination/protocol conditions. Strict TSR and contract-qualified cTSR are separate metrics and are not substituted for ordinary TSR. Ordinary TSR alone does not establish complete authorization safety. See [`tsr_case`](model/scoring/rq2_metrics.py#L194-L325).

### Workflow references and matching

HF1 preserves edge direction but ignores duplicate handoffs and their global order. For reference and observed edge sets `E` and `O`, precision is `n(E ∩ O) / n(O)` and recall is `n(E ∩ O) / n(E)`; HF1 is their harmonic mean.

ASR forms adjacent directed pairs from each role sequence and counts their multiplicities. The matched count is `m = sum_t min(observed_count(t), reference_count(t))`. Precision divides `m` by the total observed transitions, recall divides it by the total reference transitions, and ASR is their harmonic mean. It measures adjacent-transition agreement, not complete sequence equality or attack success rate.

When multiple references are applicable, HF1 takes its best match among the fixed reference edge set and legal-trajectory edge sets; ASR takes its best match among the fixed legal trajectories. Each metric selects its own best match. Compute scores per case, then average eligible normal runs; do not pool edges or transitions across cases before calculating F1. Missing workflow references are excluded and their counts reported. See [`hf1_from_edges`, `asr_from_trajectories`, and `workflow_case`](model/scoring/rq2_metrics.py#L352-L440).

### Failures, repeated rounds, and exported summaries

Report the assigned, normal, failed, and reference-eligible counts with each comparison. In the separate all-assigned view, Accuracy uses `A` and TSR uses the fixed reference-eligible assignments `T`; failed runs contribute no success. For all-assigned classification, retain each failed run's reference label and treat its prediction as missing, never as `BLOCK`. For FER, report failures among reference non-execution cases separately as unresolved outcomes, not as evidence of safe decisions.

Calculate every metric within each round first, then use an equal-weight mean across rounds. When reporting dispersion, use the sample standard deviation (`ddof=1`); a single round has no estimated standard deviation. Report the effective number of rounds when a metric is unavailable in some rounds. Keep datasets, backbones, methods, and reference versions separate; do not silently pool them or mix normal-completion and all-assigned denominators.

The minimal runner's [`report.py`](model/scoring/report.py) aggregates the records supplied to it: unfiltered output includes failed runs in classification/TSR and any available workflow scores. Use the per-case records to apply the comparison filters and fixed reference sets above; raw aggregate fields are not automatically the normal-completion comparison view. Its case-bootstrap confidence intervals are also distinct from the across-round sample standard deviation.

## Attribution and licensing

TAPay adds structured intent, transaction–intent alignment, dynamic decision boundaries, an offline payment environment and evaluation utilities to an adapted hierarchical multi-agent implementation.

The collaboration implementation builds on **TalkHier**, “Talk Structurally, Act Hierarchically: A Collaborative Framework for LLM Multi-Agent Systems,” by Zhao Wang, Sota Moriyama, Wei-Yao Wang, Briti Gangopadhyay and Shingo Takamatsu (https://arxiv.org/abs/2502.11098). The upstream distribution supplies Creative Commons Attribution–NonCommercial 4.0 International. Its original license text is retained in [model/LICENSE-TalkHier.txt](model/LICENSE-TalkHier.txt) and applies to upstream/derived material. The adapted implementation includes changes to orchestration, structured-output handling, execution controls and audit traces. No endorsement by upstream authors is implied.

This is a minimal source-and-data release, not the full experiment workspace. It excludes literature-baseline implementations, historical runs, raw trajectories, local credentials, provider endpoints, review work files and paper drafts. It therefore does not independently reproduce every comparison table in the manuscript. The mock path uses scripted, label-aware responses for software validation only, never for reporting LLM performance.

No additional license is granted here for original TAPay additions or datasets; their reuse licensing is pending an explicit decision by the rights holder. Public repository access does not override the upstream noncommercial terms.

The model implementation is synchronized with the RQ1/RQ2 experiment snapshots as of October 14, 2026. `model/SOURCE_MANIFEST.json` records source/release hashes. Packaging changes relocate `src` to `model`, update code hashing to that path, add a public command wrapper, and translate code comments/docstrings into English. Internal h1/h2/h3 names are retained for code compatibility and mapped to public Agent names in the README.
