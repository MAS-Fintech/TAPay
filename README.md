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
├── datasets/                 # 311 master, 55 difficult, 48 holdout cases
├── model/                    # Core agents, simulation, orchestration, scoring
├── config/                   # Runtime settings and credential-free example
├── main.py                   # Run TAPay or a component ablation
├── verify_release.py         # Verify packaged datasets
├── requirements.txt
├── README.md
├── NOTICE.md
└── LICENSE-TalkHier.txt
```

## Install

Python 3.11 is recommended. From this directory:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_release.py
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

TAPay adapts the TalkHier collaboration framework. See [NOTICE.md](NOTICE.md) and [LICENSE-TalkHier.txt](LICENSE-TalkHier.txt) for upstream attribution, noncommercial terms, packaging scope and the pending license decision for original additions/data. No credentials, environment folders, caches or historical experiment logs are included.
