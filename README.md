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

## Attribution and licensing

TAPay adds structured intent, transaction–intent alignment, dynamic decision boundaries, an offline payment environment and evaluation utilities to an adapted hierarchical multi-agent implementation.

The collaboration implementation builds on **TalkHier**, “Talk Structurally, Act Hierarchically: A Collaborative Framework for LLM Multi-Agent Systems,” by Zhao Wang, Sota Moriyama, Wei-Yao Wang, Briti Gangopadhyay and Shingo Takamatsu (https://arxiv.org/abs/2502.11098). The upstream distribution supplies Creative Commons Attribution–NonCommercial 4.0 International. Its original license text is retained in [model/LICENSE-TalkHier.txt](model/LICENSE-TalkHier.txt) and applies to upstream/derived material. The adapted implementation includes changes to orchestration, structured-output handling, execution controls and audit traces. No endorsement by upstream authors is implied.

This is a minimal source-and-data release, not the full experiment workspace. It excludes literature-baseline implementations, historical runs, raw trajectories, local credentials, provider endpoints, review work files and paper drafts. It therefore does not independently reproduce every comparison table in the manuscript. The mock path uses scripted, label-aware responses for software validation only, never for reporting LLM performance.

No additional license is granted here for original TAPay additions or datasets; their reuse licensing is pending an explicit decision by the rights holder. Public repository access does not override the upstream noncommercial terms.

`model/SOURCE_MANIFEST.json` records source/release hashes. Packaging changes relocate `src` to `model`, update code hashing to that path, add a public command wrapper, and translate code comments/docstrings into English. Internal h1/h2/h3 names are retained for code compatibility and mapped to public Agent names in the README.
