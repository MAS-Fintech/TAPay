"""Run aggregation: per-case metrics -> RQ2 summary with bootstrap CIs.

Aggregation rules (RQ2 spec):
  * metrics are computed per case, then aggregated over the frozen test set;
  * failure-category runs count as errors in Accuracy and are also reported
    via valid-run rate and failure counts — never rewritten into BLOCK;
  * TSR/Strict TSR exclude REFERENCE_MISSING cases from their denominator;
  * HF1/ASR are reported over cases with a reviewed workflow reference;
  * bootstrap CIs use case-level resampling with a fixed seed.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence

from scoring.rq2_metrics import classification_metrics

BOOTSTRAP_ITERS = 1000
BOOTSTRAP_SEED = 20260828


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summary_of(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    cls = classification_metrics(records)

    tsr_vals = [r["tsr"] for r in records if r.get("tsr") is not None]
    strict_vals = [r["strict_tsr"] for r in records if r.get("strict_tsr") is not None]
    hf1_vals = [r["hf1"] for r in records if r.get("hf1") is not None]
    asr_vals = [r["asr"] for r in records if r.get("asr") is not None]
    def coverage(subset):
        n_sub = len(subset)
        successes = sum(r.get("contract_tsr") is True for r in subset)
        unknown = sum(r.get("contract_tsr") is None for r in subset)
        return {"n": n_sub, "successes": successes, "unknown": unknown,
                "lower": successes/n_sub if n_sub else None,
                "upper": (successes+unknown)/n_sub if n_sub else None}
    fixed = coverage([r for r in records if r.get("contract_reference_eligible") is True])

    return {
        "n_cases": n,
        "metric_protocol": "audit-v4-units",
        "formal_acc_eligible_n": sum(r.get("formal_acc_eligible") is True for r in records),
        "primary_task_metric": "contract_fixed_reference.lower",
        "contract_fixed_reference": fixed,
        "contract_planned_bounds": coverage(records),
        "contract_expected_execute": coverage([r for r in records if r.get("gold_behavior") == "EXECUTE"]),
        "contract_predicted_execute": coverage([r for r in records if r.get("pred_behavior") == "EXECUTE"]),
        "contract_tsr_note": "conditional on verified outcomes; denominator varies by condition, not a primary comparison",
        "workflow_metric_note": "HF1: directed edge-set F1; ASR: directed adjacent-transition multiset F1, not global sequence correctness",
        "contract_tsr": (_mean([float(r["contract_tsr"]) for r in records if r.get("contract_tsr") is not None])
                         if any(r.get("contract_tsr") is not None for r in records) else None),
        "contract_tsr_verified_n": sum(r.get("contract_tsr") is not None for r in records),
        "contract_tsr_unverifiable_n": sum(r.get("contract_tsr") is None for r in records),
        "contract_tsr_successes_over_planned": sum(r.get("contract_tsr") is True for r in records) / n if n else 0.0,
        "legacy_tsr_note": "tsr/strict_tsr retained for historical comparability; not the primary verified contract metric",
        "mean_wall_seconds": _mean([r.get("wall_seconds", 0.0) for r in records]),
        "mean_tool_calls": _mean([r.get("tool_calls", 0) for r in records]),
        "mean_simulation_calls": _mean([r.get("simulation_calls", 0) for r in records]),
        "model_call_coverage_n": sum(r.get("model_invocations") is not None for r in records),
        "mean_model_invocations_observed": (_mean([r["model_invocations"] for r in records if r.get("model_invocations") is not None])
                                           if any(r.get("model_invocations") is not None for r in records) else None),
        "split_counts": {split: sum(r.get("split", "unknown") == split for r in records)
                         for split in sorted({r.get("split", "unknown") for r in records})},
        "accuracy": cls.accuracy,
        "valid_run_rate": cls.valid_run_rate,
        "failure_counts": cls.failure_counts,
        "per_class": cls.per_class,
        "macro": cls.macro,
        "weighted": cls.weighted,
        "support": cls.support,
        "predicted_counts": cls.predicted_counts,
        "tsr": _mean([float(v) for v in tsr_vals]),
        "tsr_n": len(tsr_vals),
        "strict_tsr": _mean([float(v) for v in strict_vals]),
        "hf1": _mean(hf1_vals),
        "hf1_n": len(hf1_vals),
        "asr": _mean(asr_vals),
        "asr_n": len(asr_vals),
        "reference_missing": sum(
            1 for r in records if r.get("workflow_reference") == "REFERENCE_MISSING"
        )
        + sum(1 for r in records if r.get("tsr") is None),
    }


def _metric_vector(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    s = _summary_of(records)
    return {
        "accuracy": s["accuracy"],
        "macro_f1": s["macro"]["f1"],
        "tsr": s["tsr"],
        "hf1": s["hf1"],
        "asr": s["asr"],
    }


def bootstrap_cis(
    records: Sequence[Dict[str, Any]],
    iters: int = BOOTSTRAP_ITERS,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Dict[str, float]]:
    """Case-level bootstrap 95% CIs for the headline metrics."""
    if not records:
        return {}
    rng = random.Random(seed)
    samples: Dict[str, List[float]] = {
        k: [] for k in ("accuracy", "macro_f1", "tsr", "hf1", "asr")
    }
    n = len(records)
    for _ in range(iters):
        draw = [records[rng.randrange(n)] for _ in range(n)]
        vec = _metric_vector(draw)
        for k, v in vec.items():
            samples[k].append(v)
    cis: Dict[str, Dict[str, float]] = {}
    for k, vals in samples.items():
        vals.sort()
        lo = vals[int(0.025 * iters)]
        hi = vals[min(iters - 1, int(0.975 * iters))]
        cis[k] = {"ci95_lo": lo, "ci95_hi": hi}
    return cis


def aggregate(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary = _summary_of(records)
    summary["bootstrap_ci95"] = bootstrap_cis(records)
    return summary
