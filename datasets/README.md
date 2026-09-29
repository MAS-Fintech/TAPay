# TAPay datasets

These are offline payment simulation cases, not customer transaction records or live payment outcomes. Files are copied byte-for-byte from the selected local release inputs. SHA-256 hashes and counts are recorded in `manifest.json`.

| File | Cases | Purpose | Stored splits |
|---|---:|---|---|
| `master_311.jsonl` | 311 | Sealed master collection | 283 test, 28 dev |
| `hard_55.jsonl` | 55 | Runtime-ready difficult diagnostic cases | 53 test, 2 dev |
| `holdout_48.jsonl` | 48 | Runtime-ready new-source holdout | 48 test |

The hard set is related to the master collection; these counts must not be summed as independent observations. The hard set is diagnostic, not a newly independent confirmation set. Use **`--split all`** to evaluate all 55 cases; the underlying runner otherwise defaults to `test` and selects only 53. The holdout comes from the same generation pipeline with different sources, not a real-world external dataset. `manifest.json` preserves exact inputs, rather than silently relabeling their splits.

Each record contains a conversation (`input`), simulator environment and execution policy (`runtime`), scorer-only labels and references (`evaluation`), and review metadata (`review`). Labels, review metadata and unreleased successor turns must not be exposed to the model. The included runner enforces these visibility boundaries.

The sealed master predates the gold-independent runtime policy. It is included for dataset inspection and reuse, not as the default real-provider input. Use the two runtime-ready inputs for the commands in the root README. Do not enable legacy runtime budgets to reproduce current results.

Review eligibility in the supplied files is 311/311 for the master, 35/55 for the hard set and 32/48 for the holdout. The other 20 and 16 records are labeled `AUTO_GENERATED` with `formal_acc_eligible=false`. The full runtime sets are suitable for diagnostic use; they must not be described as wholly human-confirmed. `--require-reviewed` rejects them by design.

The source metadata records review status; this release does not perform a new human annotation or certify the existing labels. The environment models authorization, intent revisions, asset/amount constraints and dynamic events under synthetic rules. Performance here is not evidence of actual bank integration, real fund settlement or production safety.

Checksums are listed in `manifest.json`. Attribution and redistribution terms are described in the root [README.md](../README.md#attribution-and-licensing); no third-party license is silently applied to the data.
