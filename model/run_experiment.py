"""Internal experiment runner for TAPay; use the root main.py entry point.

Usage (real provider):
    python src/run_experiment.py --input data/cases.jsonl --output results/<run> \
        --config-llm config/config_llm.ini

Usage (offline replay validation, no API key needed):
    python src/run_experiment.py --input data/cases.jsonl --output results/<run> --mock

The output directory must not exist unless --resume is given.
"""
from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from multiagent.llm import load_llm
from payment.cases import load_cases
from payment.hooks import build_hooks, condition_name
from payment.mock_replay import build_replay_model
from payment.runner import CaseResult, run_case
from payment.simulated_user import run_case_with_simulated_user
from experiment_integrity import (AttemptStore, failure_record, freeze_run,
                                  run_identity, select_cases)
from scoring.audit_contract import load_constraint_annotations
from scoring.report import aggregate
from scoring.rq2_metrics import per_case_metrics


def get_args():
    parser = argparse.ArgumentParser(
        description="Payment-adapted TalkHier baseline runner (B000)."
    )
    parser.add_argument("--input", required=True, help="merged cases.jsonl path")
    parser.add_argument("--output", required=True, help="empty output directory")
    parser.add_argument("--config", default="config/config.ini")
    parser.add_argument("--config-llm", default="config/config_llm.ini")
    for flag in ("h1", "h2", "h3"):
        parser.add_argument(f"--{flag}", choices=["on", "off"], default="off")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--case-index", type=int, nargs="*", default=None)
    parser.add_argument("--sample", type=int, default=-1,
                        help="randomly sample N cases from the input (seeded)")
    parser.add_argument("--seed", type=int, default=20260828,
                        help="sampling seed for --sample")
    parser.add_argument("--mock", action="store_true",
                        help="offline replay validation with scripted models")
    parser.add_argument("--allow-legacy-runtime-budget", action="store_true",
                        help="explicit historical-protocol comparison only; old quotas may depend on gold")
    parser.add_argument("--simulated-user", choices=["on", "off"], default="off",
                        help="diagnostic mode: a simulated user answers "
                        "REQUEST_CLARIFICATION questions; records how many "
                        "rounds are needed to leave the clarification state. "
                        "Standard B000 results (off) are the official baseline.")
    parser.add_argument("--max-clarification-rounds", type=int, default=3)
    parser.add_argument("--release-policy", default=None,
                        choices=["on_first_engagement", "disabled"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--split", choices=["test", "dev", "all"], default="test")
    parser.add_argument("--require-reviewed", action="store_true")
    parser.add_argument("--constraint-annotations", default=None, help="optional scorer-only sidecar; never visible to agents")
    parser.add_argument("--retry-infrastructure", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_config(path: str, llm_path: str) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(path, encoding="utf-8")
    llm_config = configparser.ConfigParser()
    llm_config.read(llm_path, encoding="utf-8")
    for section in llm_config.sections():
        if not config.has_section(section):
            config.add_section(section)
        for key, value in llm_config.items(section):
            config.set(section, key, value)
    return config


def _sha256_of_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _code_hashes(root: Path) -> dict:
    hashes = {}
    for p in sorted((root / "model").rglob("*.py")):
        hashes[str(p.relative_to(root))] = _sha256_of_file(str(p))
    return hashes


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _case_complete(case_dir: Path) -> bool:
    return (case_dir / "prediction.json").exists() and (
        case_dir / "trajectory.jsonl"
    ).exists()


def main() -> int:
    args = get_args()
    from execution_schedule import provider_lease
    config = load_config(args.config, args.config_llm)
    with provider_lease(config, mock=args.mock):
        return _main(args)


def _main(args) -> int:
    out_dir = Path(args.output)
    if out_dir.exists() and not args.resume:
        print(f"error: output directory {out_dir} already exists; "
              "pass --resume to continue it", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config, args.config_llm)
    run_cfg = config["RUN"] if config.has_section("RUN") else {}
    recursion_limit = int(run_cfg.get("recursion_limit", 30))
    release_policy = args.release_policy or run_cfg.get(
        "release_policy", "on_first_engagement"
    )
    scoring_cfg = config["SCORING"] if config.has_section("SCORING") else {}
    rel_tol = float(scoring_cfg.get("amount_rel_tol", 1e-6))
    abs_tol = float(scoring_cfg.get("amount_abs_tol", 1e-9))

    h1 = args.h1 == "on"
    h2 = args.h2 == "on"
    h3 = args.h3 == "on"
    condition = condition_name(h1, h2, h3)
    # Hooks are built per case (H1/H2/H3 carry per-case state). All eight
    # conditions B000..B111 are executable.

    def make_hooks(model):
        return build_hooks(h1=h1, h2=h2, h3=h3, llm=model)

    cases = select_cases(load_cases(args.input), args.split)
    if (not args.mock and not getattr(args, "allow_legacy_runtime_budget", False)
            and any(not c.runtime_policy for c in cases)):
        raise ValueError("Real runs require round6 runtime policy. Use scripts/prepare_round6.py "
                         "or explicitly --allow-legacy-runtime-budget for a historical comparison.")
    sampled_indices = None
    if args.case_index is not None:
        cases = [cases[i] for i in args.case_index]
        sampled_indices = list(args.case_index)
    if args.sample > 0:
        import random

        rng = random.Random(args.seed)
        all_cases = select_cases(load_cases(args.input), args.split)
        sampled_indices = sorted(rng.sample(range(len(all_cases)),
                                            min(args.sample, len(all_cases))))
        cases = [all_cases[i] for i in sampled_indices]
    if args.limit > 0:
        cases = cases[: args.limit]
        if sampled_indices is not None:
            sampled_indices = sampled_indices[: args.limit]

    cases = select_cases(cases, "all", args.require_reviewed)
    constraint_annotations = load_constraint_annotations(args.constraint_annotations, args.input)
    identity = run_identity(cases, config, _code_hashes(Path(__file__).resolve().parent.parent),
                            constraint_annotations=constraint_annotations,
                            input_sha256=_sha256_of_file(args.input), condition=condition,
                            mock=args.mock, split=args.split, require_reviewed=args.require_reviewed, release_policy=release_policy,
                            recursion_limit=recursion_limit, workers=args.workers,
                            simulated_user=args.simulated_user,
                            max_clarification_rounds=args.max_clarification_rounds,
                            max_attempts=args.max_attempts, retry_infrastructure=args.retry_infrastructure)
    identity_hash = freeze_run(out_dir, identity)
    store = AttemptStore(out_dir, identity_hash, args.retry_infrastructure, args.max_attempts)
    llm = None if args.mock else load_llm(config)
    sim_user = args.simulated_user == "on"
    if sim_user and args.mock:
        print("error: --simulated-user requires a real LLM (the replay mock "
              "never asks for clarification)", file=sys.stderr)
        return 2

    def execute(case) -> CaseResult:
        model = build_replay_model(case, h1_intent="auto" if h1 else None) if args.mock else llm
        return run_case(
            case,
            model,
            make_hooks(model),
            release_policy=release_policy,
            recursion_limit=recursion_limit,
            verbose=args.verbose,
        )

    def process(case):
        case_dir = out_dir / "cases" / case.case_id
        attempt = store.begin("", case.case_id)
        if attempt is None:
            return store.record("", case.case_id)
        attempt_started = time.monotonic()
        store.started(attempt)
        try:
            if sim_user:
                sim_result = run_case_with_simulated_user(
                    case,
                    llm,
                    make_hooks(llm),
                    max_rounds=args.max_clarification_rounds,
                    release_policy=release_policy,
                    recursion_limit=recursion_limit,
                    verbose=args.verbose,
                )
                result = sim_result.final_result
                record = result.prediction_record()
                record["simulated_user"] = {
                    "initial_behavior": sim_result.initial_behavior,
                    "final_behavior": sim_result.final_behavior,
                    "clarification_rounds": sim_result.clarification_rounds,
                    "recovered": sim_result.recovered,
                    "round_history": sim_result.round_history,
                }
            else:
                result = execute(case)
                record = result.prediction_record()
            store.finish(attempt, record, result.raw_trajectory)
        except Exception as exc:
            record = failure_record(case.case_id, f"driver_error: {type(exc).__name__}: {exc}",
                                    time.monotonic() - attempt_started)
            store.finish(attempt, record)
        extra = ""
        if sim_user and "simulated_user" in record:
            extra = (f" rounds={record['simulated_user']['clarification_rounds']}"
                     f" initial={record['simulated_user']['initial_behavior']}")
        print(f"[{case.case_id}] behavior={record['final_behavior']} "
              f"failure={record['failure']} wall={record['wall_seconds']}s{extra}")
        return record

    started = time.time()
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            records = list(pool.map(process, cases))
    else:
        records = [process(c) for c in cases]

    per_case = []
    # Zip cases with their in-memory records by position: multiple pathology
    # variants can share one case_id, so re-reading case_dir/prediction.json
    # per case would alias every variant to the last one written to disk.
    for case, record in zip(cases, records):
        result = CaseResult(
            case_id=case.case_id,
            prediction=_prediction_from_record(record),
            raw_trajectory=[],
            agent_trajectory=record["agent_trajectory"],
            handoff_sequence=record["handoff_sequence"],
            receipts=record.get("receipts", []),
            tool_log=record.get("tool_log", []),
            released_events=record.get("released_events", []),
            final_state_version=record.get("final_state_version", 1),
            sim_calls=record.get("sim_calls", 0),
            wall_seconds=record.get("wall_seconds", 0.0),
            execution_context=record.get("execution_context"),
            model_call_audit=record.get("model_call_audit", {}),
        )
        row = per_case_metrics(case, result, rel_tol=rel_tol, abs_tol=abs_tol,
                               constraint_annotations=constraint_annotations.get(case.case_id))
        if sim_user:
            # Per user instruction, the simulated-user diagnostic does not
            # compute TSR/ASR; it focuses on behavior flow + clarification
            # rounds.  HF1 stays as the workflow-coverage readout.
            sim = record.get("simulated_user", {})
            row["tsr"] = None
            row["strict_tsr"] = None
            row["contract_tsr"] = None
            row["contract_unverifiable"] = ["simulated-user diagnostic only"]
            row["asr"] = None
            row["initial_behavior"] = sim.get("initial_behavior")
            row["clarification_rounds"] = sim.get("clarification_rounds", 0)
            row["recovered"] = sim.get("recovered", False)
        per_case.append(row)

    with open(out_dir / "metrics_per_case.jsonl", "w", encoding="utf-8") as f:
        for row in per_case:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = aggregate(per_case)
    summary["attempt_accounting"] = store.stats("", cases)
    print(f"contract_tsr={summary['contract_tsr']} verified={summary['contract_tsr_verified_n']}/{len(cases)}; unknown={summary['contract_tsr_unverifiable_n']}")
    if sim_user:
        clarified = [
            r for r in per_case if r.get("initial_behavior") == "REQUEST_CLARIFICATION"
        ]
        rounds_list = [r["clarification_rounds"] for r in clarified]
        summary["simulated_user_diagnostic"] = {
            "note": "TSR/ASR intentionally not computed in simulated-user mode.",
            "cases_initially_request_clarification": len(clarified),
            "recovered_after_simulated_answers": sum(
                1 for r in clarified if r.get("recovered")
            ),
            "still_clarification": sum(
                1
                for r in clarified
                if r.get("pred_behavior") == "REQUEST_CLARIFICATION"
            ),
            "mean_rounds_among_clarified": (
                sum(rounds_list) / len(rounds_list) if rounds_list else 0.0
            ),
            "max_rounds_among_clarified": max(rounds_list) if rounds_list else 0,
        }
    _write_json(out_dir / "metrics_summary.json", summary)

    config_snapshot = {s: dict(config.items(s)) for s in config.sections()}
    for _sec in config_snapshot.values():
        for _k in list(_sec):
            if "key" in _k.lower() and _sec[_k]:
                _sec[_k] = "***"
    manifest = {
        "run_identity_sha256": identity_hash,
        "run_name": out_dir.name,
        "condition": condition,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "mock": bool(args.mock),
        "simulated_user": sim_user,
        "config_llm_path": args.config_llm,
        "max_clarification_rounds": args.max_clarification_rounds if sim_user else 0,
        "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(time.time() - started, 3),
        "input_path": os.path.abspath(args.input),
        "input_sha256": identity["input_sha256"],
        "n_cases": len(cases),
        "sample_seed": args.seed if args.sample > 0 else None,
        "sampled_indices": sampled_indices,
        "config": config_snapshot,
        "code_sha256": identity["code_sha256"],
        "release_policy": release_policy,
        "tolerances": {"rel_tol": rel_tol, "abs_tol": abs_tol},
        "scope_note": (
            "Offline payment simulation only; this run makes no statement "
            "about real on-chain safety, production payment security, fraud "
            "detection, or regulatory compliance."
        ),
    }
    if not (out_dir / "manifest.json").exists():
        _write_json(out_dir / "manifest.json", manifest)
    _write_json(out_dir / "last_invocation.json", manifest)

    print(f"\n== {condition} summary over {summary['n_cases']} cases ==")
    print(f"accuracy={summary['accuracy']:.4f} macro_f1={summary['macro']['f1']:.4f} "
          f"legacy_tsr={summary['tsr']:.4f} hf1={summary['hf1']:.4f} asr={summary['asr']:.4f} "
          f"valid_run_rate={summary['valid_run_rate']:.4f}")
    if sim_user:
        d = summary["simulated_user_diagnostic"]
        print(f"simulated-user diagnostic: initially_clarify={d['cases_initially_request_clarification']} "
              f"recovered={d['recovered_after_simulated_answers']} "
              f"mean_rounds={d['mean_rounds_among_clarified']:.2f} "
              f"(TSR/ASR intentionally skipped in this mode)")
    return 0


def _prediction_from_record(record):
    from payment.prediction import Prediction

    return Prediction(
        final_behavior=record.get("final_behavior"),
        action_id=record.get("action_id"),
        operation=record.get("operation"),
        parameters=record.get("parameters"),
        receipt_id=record.get("receipt_id"),
        rationale=record.get("rationale"),
        receipt=record.get("receipt"),
        legal_finish=record.get("legal_finish", False),
        failure=record.get("failure"),
        failure_detail=record.get("failure_detail", ""),
    )


if __name__ == "__main__":
    sys.exit(main())
