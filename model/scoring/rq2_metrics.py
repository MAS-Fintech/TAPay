"""RQ2 metric implementations for the Payment-adapted TalkHier baseline.

Implements the seven primary metrics exactly as specified in the project's
RQ2 spec (docs/METRICS.md): Accuracy, Precision, Recall, F1 (per-class,
macro, weighted), TSR (task success against the reviewed task contract),
HF1 (unordered directed handoff-edge F1) and ASR (ordered transition
multiset F1), plus the Strict TSR audit variant.

Case-level API; aggregation (incl. bootstrap CIs) lives in scoring/report.py.
REFERENCE_MISSING is returned whenever the required reviewed reference is
absent; such cases never enter a denominator and are never guessed.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from payment.cases import PaymentCase
from payment.prediction import BEHAVIORS, Prediction
from payment.runner import CaseResult

REFERENCE_MISSING = "REFERENCE_MISSING"

DEFAULT_REL_TOL = 1e-6
DEFAULT_ABS_TOL = 1e-9
DEADLINE_TOLERANCE_SECONDS = 60.0


# ---------------------------------------------------------------------------
# classification metrics (Accuracy / Precision / Recall / F1)
# ---------------------------------------------------------------------------

@dataclass
class ClassMetrics:
    per_class: Dict[str, Dict[str, float]]
    macro: Dict[str, float]
    weighted: Dict[str, float]
    accuracy: float
    support: Dict[str, int]
    predicted_counts: Dict[str, int]
    valid_run_rate: float
    failure_counts: Dict[str, int]
    n: int


def classification_metrics(records: Sequence[Dict[str, Any]]) -> ClassMetrics:
    """records: [{gold_behavior, pred_behavior (None on failure), failure}]."""
    n = len(records)
    failure_counts: Dict[str, int] = {}
    support = {b: 0 for b in BEHAVIORS}
    predicted_counts = {b: 0 for b in BEHAVIORS}
    correct = 0
    valid = 0
    for rec in records:
        gold = rec["gold_behavior"]
        pred = rec.get("pred_behavior")
        if gold in support:
            support[gold] += 1
        if rec.get("failure"):
            failure_counts[rec["failure"]] = failure_counts.get(rec["failure"], 0) + 1
        else:
            valid += 1
        if pred in predicted_counts:
            predicted_counts[pred] += 1
        if pred is not None and pred == gold:
            correct += 1

    per_class: Dict[str, Dict[str, float]] = {}
    precs, recs, f1s = [], [], []
    for c in BEHAVIORS:
        tp = sum(
            1
            for r in records
            if r.get("pred_behavior") == c and r["gold_behavior"] == c
        )
        fp = predicted_counts[c] - tp
        fn = support[c] - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        per_class[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        precs.append(precision)
        recs.append(recall)
        f1s.append(f1)

    total = sum(support.values()) or 1
    macro = {
        "precision": sum(precs) / len(BEHAVIORS),
        "recall": sum(recs) / len(BEHAVIORS),
        "f1": sum(f1s) / len(BEHAVIORS),
    }
    weighted = {
        "precision": sum(
            per_class[c]["precision"] * support[c] for c in BEHAVIORS
        )
        / total,
        "recall": sum(per_class[c]["recall"] * support[c] for c in BEHAVIORS) / total,
        "f1": sum(per_class[c]["f1"] * support[c] for c in BEHAVIORS) / total,
    }
    return ClassMetrics(
        per_class=per_class,
        macro=macro,
        weighted=weighted,
        accuracy=correct / n if n else 0.0,
        support=support,
        predicted_counts=predicted_counts,
        valid_run_rate=valid / n if n else 0.0,
        failure_counts=failure_counts,
        n=n,
    )


# ---------------------------------------------------------------------------
# TSR
# ---------------------------------------------------------------------------

@dataclass
class TsrOutcome:
    status: Optional[bool]  # None == REFERENCE_MISSING
    strict: Optional[bool]
    reasons: List[str] = field(default_factory=list)
    protocol_reasons: List[str] = field(default_factory=list)


def _num_close(a: Any, b: Any, rel_tol: float, abs_tol: float) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    except (TypeError, ValueError):
        return False


def _deadline_close(a: Any, b: Any, tolerance_seconds: float) -> bool:
    try:
        da = datetime.fromisoformat(str(a))
        db = datetime.fromisoformat(str(b))
        return abs((da - db).total_seconds()) <= tolerance_seconds
    except (TypeError, ValueError):
        return False


def _field_value(prediction: Prediction, name: str) -> Any:
    if prediction.parameters and name in prediction.parameters:
        return prediction.parameters[name]
    if name == "operation":
        return prediction.operation
    return getattr(prediction, name, None)


def _contract_field_matches(
    name: str, expected: Any, actual: Any, rel_tol: float, abs_tol: float
) -> bool:
    if name in ("amount", "slippage", "time_window_seconds"):
        return _num_close(actual, expected, rel_tol, abs_tol)
    if name == "deadline_at":
        return _deadline_close(actual, expected, DEADLINE_TOLERANCE_SECONDS)
    return actual == expected


def _receipt_param_match(
    receipt: Dict[str, Any],
    parameters: Dict[str, Any],
    catalog_param_names: Sequence[str],
    rel_tol: float,
    abs_tol: float,
) -> bool:
    rparams = receipt.get("parameters", {})
    for name in catalog_param_names:
        if name not in parameters:
            continue  # parameter absent from the final decision: contract governs
        expected = parameters[name]
        got = rparams.get(name)
        if isinstance(expected, (int, float)) or isinstance(got, (int, float)):
            if not _num_close(got, expected, rel_tol, abs_tol):
                return False
        elif got != expected:
            return False
    return True


def tsr_case(
    case: PaymentCase,
    result: CaseResult,
    *,
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float = DEFAULT_ABS_TOL,
) -> TsrOutcome:
    gold = case.gold
    task_ref = case.task_reference
    expected_behavior = gold.get("expected_behavior")
    contract = task_ref.get("requested_action_contract")
    if not expected_behavior or not isinstance(contract, dict):
        return TsrOutcome(status=None, strict=None, reasons=[REFERENCE_MISSING])

    pred = result.prediction
    reasons: List[str] = []
    strict_reasons: List[str] = []

    if pred.failure is not None:
        return TsrOutcome(
            status=False, strict=False, reasons=[f"run failure: {pred.failure}"]
        )
    if pred.final_behavior != expected_behavior:
        reasons.append(
            f"behavior mismatch: predicted {pred.final_behavior}, "
            f"expected {expected_behavior}"
        )

    successful_receipts = [
        r for r in result.receipts if r.get("status") == "SIMULATED_EXECUTED"
    ]

    if expected_behavior == "EXECUTE":
        runtime_action_ids = (task_ref.get("runtime_action_id_map") or {}).values()
        allowed_action_ids = set(runtime_action_ids)
        if pred.action_id not in allowed_action_ids:
            reasons.append(
                f"action_id {pred.action_id!r} not in allowed runtime actions"
            )
        if pred.operation not in (task_ref.get("allowed_operations") or []):
            reasons.append(f"operation {pred.operation!r} not allowed")

        field_status = contract.get("field_status", {})
        for name, status in field_status.items():
            if status != "PRESENT_VALID":
                continue
            expected = contract.get(name)
            actual = _field_value(pred, name)
            if actual is None:
                reasons.append(f"contract field {name} missing from prediction")
            elif not _contract_field_matches(name, expected, actual, rel_tol, abs_tol):
                reasons.append(
                    f"contract field {name}: predicted {actual!r} != {expected!r}"
                )

        catalog_param_names: List[str] = []
        # catalog parameter names come from the current action catalog
        # (v4 keeps it at the environment top level; v3 nests it in
        # initial_state).
        catalog = (
            case.environment["initial_state"].get("action_catalog")
            or case.environment.get("action_catalog", [])
        )
        chosen = next(
            (a for a in catalog if a.get("id") == pred.action_id), None
        )
        if chosen:
            catalog_param_names = list(chosen.get("parameters", {}).keys())

        def receipt_ok(r: Dict[str, Any], require_cited: bool) -> bool:
            if r.get("action_id") != pred.action_id:
                return False
            if r.get("state_version") != result.final_state_version:
                return False
            if require_cited and r.get("receipt_id") != pred.receipt_id:
                return False
            if isinstance(pred.parameters, dict) and not _receipt_param_match(
                r, pred.parameters, catalog_param_names, rel_tol, abs_tol
            ):
                return False
            return True

        cited = [r for r in successful_receipts if receipt_ok(r, True)]
        matched = [r for r in successful_receipts if receipt_ok(r, False)]
        if not matched:
            reasons.append(
                "no successful receipt bound to the chosen transaction on the "
                "final environment state"
            )
        if not cited:
            strict_reasons.append(
                "no successful receipt explicitly cited by the prediction"
            )

        sim_max = case.simulation_policy.get("simulation_max_calls")
        if sim_max is not None and result.sim_calls > sim_max:
            reasons.append(
                f"simulation budget exceeded: {result.sim_calls} > {sim_max}"
            )
            strict_reasons.append("simulation budget exceeded")
        if not pred.legal_finish:
            strict_reasons.append("no legal FINISH")
        if _termination_violation(result.agent_trajectory):
            strict_reasons.append(
                "repeated evaluator contact without new proposal or evidence"
            )
    else:
        # BLOCK / REQUEST_CLARIFICATION: a successful execution violates the
        # decision ONLY when the final decision itself cites it
        # (final_decision.receipt_id resolves to a SIMULATED_EXECUTED
        # receipt).  Successful receipts the decision does not reference are
        # due-diligence probes ("simulate first, then correctly BLOCK"), not
        # disallowed executions.
        cited_successful = [
            r
            for r in successful_receipts
            if pred.receipt_id and r.get("receipt_id") == pred.receipt_id
        ]
        if cited_successful:
            reasons.append(
                "final decision cites a successful simulation receipt"
            )
            strict_reasons.append(
                "final decision cites a successful simulation receipt"
            )
        if not pred.legal_finish:
            strict_reasons.append("no legal FINISH")

    status = not reasons
    strict = status and not strict_reasons
    return TsrOutcome(status=status, strict=strict, reasons=reasons + strict_reasons,
                      protocol_reasons=strict_reasons)


def _termination_violation(trajectory: Sequence[str]) -> bool:
    """Structural check of the dataset loop bound: an evaluator may only be
    re-contacted after a new proposal (Generator/Revisor ran in between)."""
    from payment.prompts import EVAL_MEMBERS, GENERATOR, REVISOR

    seen: Dict[str, int] = {}
    for i, node in enumerate(trajectory):
        if node in EVAL_MEMBERS:
            if node in seen:
                between = trajectory[seen[node] + 1 : i]
                if GENERATOR not in between and REVISOR not in between:
                    return True
            seen[node] = i
    return False


# ---------------------------------------------------------------------------
# HF1 / ASR
# ---------------------------------------------------------------------------

def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def hf1_from_edges(
    expected: Sequence[Sequence[str]], observed: Sequence[Sequence[str]]
) -> Dict[str, float]:
    e = {tuple(x) for x in expected}
    o = {tuple(x) for x in observed}
    if not e and not o:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    inter = len(e & o)
    p = inter / len(o) if o else 0.0
    r = inter / len(e) if e else 0.0
    return {"precision": p, "recall": r, "f1": _f1(p, r)}


def asr_from_trajectories(
    expected: Sequence[str], observed: Sequence[str]
) -> Dict[str, float]:
    be = Counter(zip(expected, expected[1:]))
    bo = Counter(zip(observed, observed[1:]))
    if not be and not bo:
        return {"transition_recall": 1.0, "transition_precision": 1.0, "asr": 1.0}
    m = sum(min(be[t], bo[t]) for t in be.keys() & bo.keys())
    tr = m / sum(be.values()) if be else 0.0
    tp = m / sum(bo.values()) if bo else 0.0
    return {"transition_recall": tr, "transition_precision": tp, "asr": _f1(tr, tp)}


@dataclass
class WorkflowOutcome:
    hf1: Optional[float]
    asr: Optional[float]
    matched_trajectory: Optional[List[str]]
    missing: bool = False


# Per-baseline static reference workflows (literature baselines): the frozen
# cases.jsonl only embeds workflow_reference.B000, so other baselines fall
# back to data/workflow_refs/<workflow_key>.json (docs/RQ1_BASELINES.md).
_WORKFLOW_REFS_DIR = Path(__file__).resolve().parents[2] / "data" / "workflow_refs"
_ref_cache: Dict[str, Optional[Dict[str, Any]]] = {}


def _load_workflow_ref(
    workflow_key: str, refs_dir: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    base = Path(refs_dir) if refs_dir is not None else _WORKFLOW_REFS_DIR
    cache_key = f"{base}|{workflow_key}"
    if cache_key not in _ref_cache:
        path = base / f"{workflow_key}.json"
        ref: Optional[Dict[str, Any]] = None
        if path.exists():
            with open(path, encoding="utf-8") as f:
                ref = json.load(f)
        _ref_cache[cache_key] = ref
    return _ref_cache[cache_key]


def workflow_case(
    case: PaymentCase,
    result: CaseResult,
    *,
    workflow_key: str = "B000",
    refs_dir: Optional[Path] = None,
) -> WorkflowOutcome:
    wf = (case.workflow_reference or {}).get(workflow_key)
    if not wf:
        wf = _load_workflow_ref(workflow_key, refs_dir)
    if not wf:
        return WorkflowOutcome(None, None, None, missing=True)
    expected_edges = wf.get("expected_handoff_edges")
    legal = wf.get("legal_trajectories")
    if not expected_edges or not legal:
        return WorkflowOutcome(None, None, None, missing=True)

    # HF1 takes the max over the workflow's legal trajectory variants, the
    # same convention ASR already uses: an elimination or regeneration
    # variant must not be structurally penalised for missing full-graph edges.
    best_hf1 = hf1_from_edges(expected_edges, result.handoff_sequence)["f1"]
    for traj in legal:
        edges = [list(e) for e in dict.fromkeys(zip(traj, traj[1:]))]
        if not edges:
            continue
        best_hf1 = max(best_hf1, hf1_from_edges(edges, result.handoff_sequence)["f1"])
    hf1 = {"f1": best_hf1}
    best_asr = -1.0
    best_traj: Optional[List[str]] = None
    for traj in legal:
        score = asr_from_trajectories(traj, result.agent_trajectory)["asr"]
        if score > best_asr:
            best_asr = score
            best_traj = list(traj)
    return WorkflowOutcome(hf1["f1"], best_asr, best_traj)


def per_case_metrics(
    case: PaymentCase,
    result: CaseResult,
    *,
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float = DEFAULT_ABS_TOL,
    constraint_annotations=None,
    workflow_key: str = "B000",
) -> Dict[str, Any]:
    tsr = tsr_case(case, result, rel_tol=rel_tol, abs_tol=abs_tol)
    wf = workflow_case(case, result, workflow_key=workflow_key)
    pred = result.prediction
    from scoring.audit_contract import audit_contract
    audited = audit_contract(case, result, rel_tol, abs_tol, constraint_annotations)
    return {
        **audited,
        "legacy_tsr": tsr.status,
        "split": case.split,
        "formal_acc_eligible": case.review.get("formal_acc_eligible", False),
        "wall_seconds": result.wall_seconds,
        "tool_calls": len(result.tool_log),
        "simulation_calls": result.sim_calls,
        "model_invocations": (len(result.model_call_audit.get("calls", []))
                              if result.model_call_audit.get("instrumented") else None),
        "model_call_wall_seconds": (sum(c.get("wall_seconds", 0) for c in result.model_call_audit.get("calls", []))
                                    if result.model_call_audit.get("instrumented") else None),
        "case_id": case.case_id,
        "dataset_kind": case.dataset_kind,
        "gold_behavior": case.gold.get("expected_behavior"),
        "pred_behavior": pred.final_behavior,
        "failure": pred.failure,
        "correct": (
            pred.failure is None
            and pred.final_behavior is not None
            and pred.final_behavior == case.gold.get("expected_behavior")
        ),
        "tsr": tsr.status,
        "strict_tsr": tsr.strict,
        "tsr_reasons": tsr.reasons,
        "hf1": wf.hf1,
        "asr": wf.asr,
        "workflow_reference": REFERENCE_MISSING if wf.missing else "OK",
    }
