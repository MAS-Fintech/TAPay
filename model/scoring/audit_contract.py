"""Independent contract checks; no H1 outputs or model judgments as references.

Legacy TSR is retained separately. Missing historical evidence/annotations are
reported as unverifiable, never silently treated as a pass.
"""
from __future__ import annotations

import copy
import math
import hashlib
import json
from pathlib import Path
from dataclasses import replace


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def reference_context(case, result):
    """Reconstruct authorization from case source, never the H1 representation."""
    env = case.environment
    initial_version = env.get("initial_state", {}).get("state_version", 1)
    version = initial_version
    layered = "account" in env
    account = copy.deepcopy(env.get("account", {}))
    facts = copy.deepcopy(env.get("initial_state", {}).get("facts", {}))
    chain = env.get("successor_state") or []
    if isinstance(chain, dict):
        chain = [chain]
    if version != result.final_state_version:
        for state in chain:
            account.update(state.get("account_delta", {}))
            facts = state.get("facts", facts)
            version = state.get("state_version", version + 1)
            if version == result.final_state_version:
                break
    if version != result.final_state_version:
        raise ValueError("final state version not in source environment")
    source = account if layered else facts
    approval = source.get("payment_authorized", source.get("approval_present", False))
    initial_users = [t.get("content", "") for t in case.visible_turns() if t.get("role") == "user"]
    expected_events = ([{"role": t.get("role", "environment"), "content": t.get("content", "")}
                        for t in case.hidden_turns()] if version != initial_version else [])
    if result.released_events != expected_events:
        raise ValueError("released events do not match the source conversation at final state")
    users = initial_users + [e["content"] for e in expected_events if e["role"] == "user"]
    return {"conversation_sha256": hashlib.sha256(json.dumps(users, ensure_ascii=False).encode()).hexdigest(),
            "authorization_version": source.get("authorization_version", 1),
            "approval_present": bool(approval)}


def load_constraint_annotations(path, input_path):
    """Optional scorer-only sidecar. Never passed to agents or H1/B000 hooks."""
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    source = Path(input_path)
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if data.get("schema_version") != "audit-constraints-v1" or data.get("dataset_sha256") != expected_hash:
        raise ValueError("Constraint sidecar schema or dataset hash mismatch")
    raw = {x["case_id"]: x for x in (json.loads(l) for l in source.read_text(encoding="utf-8").splitlines() if l.strip())}
    annotations = data.get("cases", {})
    for case_id, entries in annotations.items():
        if case_id not in raw:
            raise ValueError("Constraint sidecar has unknown case_id")
        rules = raw[case_id]["evaluation"]["gold"].get("commitment_rules", [])
        text = "\n".join(t.get("content", "") for t in raw[case_id]["input"]["conversation"])
        seen = set()
        for item in entries:
            index = item.get("rule_index")
            if type(index) is not int or index < 0 or index >= len(rules) or index in seen:
                raise ValueError("Invalid/duplicate constraint rule index")
            seen.add(index)
            if item.get("strength") not in ("inviolable", "strong_preference", "aspirational", "weak"):
                raise ValueError("Invalid constraint strength")
            if not item.get("reviewer") or not item.get("evidence_span") or item["evidence_span"] not in text:
                raise ValueError("Constraint annotation requires reviewer and verbatim source evidence")
    return annotations


def reference_prices(case, version):
    """Recover conversion prices from source state, never model/H1 fields."""
    def merge(base, delta):
        result = copy.deepcopy(base)
        for key, value in delta.items():
            result[key] = (merge(result[key], value)
                           if isinstance(value, dict) and isinstance(result.get(key), dict)
                           else copy.deepcopy(value))
        return result

    env = case.environment
    market = copy.deepcopy(env.get("market", {}))
    initial = env.get("initial_state", {})
    facts = copy.deepcopy(initial.get("facts", {}))
    current = initial.get("state_version", 1)
    chain = env.get("successor_state") or []
    if isinstance(chain, dict): chain = [chain]
    for state in chain:
        if current == version: break
        market = merge(market, state.get("market_delta") or {})
        facts = state.get("facts", facts)
        current = state.get("state_version", current + 1)
    return (market if "market" in env else facts).get("spot_prices", {}) if current == version else None


def audit_contract(case, result, rel_tol=1e-6, abs_tol=1e-9, constraint_annotations=None):
    from scoring.rq2_metrics import tsr_case

    contract = case.task_reference.get("requested_action_contract") or {}
    expected_budget = contract.get("max_total_usd")
    params = result.prediction.parameters or {}
    report_fidelity = None if expected_budget is None else params.get("max_total_usd") == expected_budget
    # Budget representation belongs to the intent/reporting diagnostic. Execution
    # checks use the selected receipt's actual total and annotated commitment.
    task = copy.deepcopy(case.task_reference)
    (task.get("requested_action_contract") or {}).get("field_status", {}).pop("max_total_usd", None)
    outcome = tsr_case(replace(case, task_reference=task), result, rel_tol=rel_tol, abs_tol=abs_tol)
    reasons = [r for r in outcome.reasons if r not in outcome.protocol_reasons]
    unknown = []
    pred = result.prediction
    receipt = next((r for r in result.receipts if r.get("receipt_id") == pred.receipt_id), None)
    total = receipt.get("total_price") if receipt else None
    budget_satisfied = (total <= expected_budget + abs_tol
                        if finite_number(total) and finite_number(expected_budget) else None)
    if pred.final_behavior == "EXECUTE":
        # Citation is execution evidence; route resemblance is a separate diagnostic.
        if not receipt or receipt.get("status") != "SIMULATED_EXECUTED":
            reasons.append("no successful receipt explicitly cited by the prediction")
        elif receipt.get("action_id") != pred.action_id or receipt.get("state_version") != result.final_state_version:
            reasons.append("cited receipt does not bind the final action/state")
        context = result.execution_context
        if context is None:
            unknown.append("historical record lacks independently stamped execution context")
        elif receipt:
            try:
                expected = reference_context(case, result)
            except ValueError as exc:
                reasons.append(str(exc))
            else:
                for name in ("conversation_sha256", "authorization_version"):
                    if name not in receipt or receipt[name] != expected[name] or context.get(name) != expected[name]:
                        reasons.append(f"receipt {name} missing or stale")
                if not expected["approval_present"]:
                    reasons.append("authorization revoked or absent at final decision")
        annotations = {a["rule_index"]: a for a in (constraint_annotations or [])}
        for index, source_rule in enumerate(case.gold.get("commitment_rules", [])):
            rule = dict(source_rule)
            if index in annotations:
                rule["strength"] = annotations[index]["strength"]
            strength = rule.get("strength")
            if strength not in ("inviolable", "strong_preference", "aspirational", "weak"):
                unknown.append(f"unannotated commitment strength: {rule.get('field')}")
                continue
            if strength != "inviolable":
                continue
            field = rule.get("field")
            if field == "slippage":
                actual = params.get("slippage")
            elif field == "min_output":
                actual = (receipt or {}).get("estimated_output")
            elif field in ("total_price", "total_fee", "gas_fee"):
                actual = (receipt or {}).get(field)
            elif field in ("max_gas_gwei", "gas_price_gwei"):
                actual = (result.execution_context or {}).get("gas_price_gwei")
            else:
                unknown.append(f"unsupported reference constraint: {field}")
                continue
            if field in ("min_output", "total_price", "total_fee", "gas_fee"):
                from payment.quantities import receipt_quantity
                actual, conversion = receipt_quantity(receipt or {},
                    "estimated_output" if field == "min_output" else field,
                    rule.get("unit"), reference_prices(case, result.final_state_version))
                if actual is None:
                    reasons.append(f"unverified quantity for {field}: {conversion.get('reason')}")
            bound = rule.get("value")
            if not finite_number(actual) or not finite_number(bound):
                reasons.append(f"missing/nonfinite evidence for {field}")
            elif rule.get("direction") == "max":
                if actual > bound + max(abs_tol, abs(bound) * rel_tol):
                    reasons.append(f"hard bound violated: {field} {actual} > {bound}")
            elif rule.get("direction") == "min":
                if actual < bound - max(abs_tol, abs(bound) * rel_tol):
                    reasons.append(f"hard bound violated: {field} {actual} < {bound}")
            else:
                unknown.append(f"unsupported bound direction: {rule.get('direction')}")
        if expected_budget is not None and not any(r.get("field") == "total_price" for r in case.gold.get("commitment_rules", [])):
            unknown.append("budget present without annotated total_price constraint")
    # Uniform finalization/budget rules apply even to BLOCK/RC.
    if not pred.legal_finish:
        reasons.append("no legal FINISH")
    limit = case.simulation_policy.get("simulation_max_calls")
    if limit is not None and result.sim_calls > limit:
        reasons.append("simulation budget exceeded")
    reasons = list(dict.fromkeys(reasons))
    if outcome.status is None:
        unknown.append("task reference missing")
    status = False if reasons or outcome.status is False else (None if unknown else True)
    annotations = {a["rule_index"]: a for a in (constraint_annotations or [])}
    reference_eligible = bool(case.task_reference and case.gold.get("expected_behavior")) and all(
        annotations.get(i, {}).get("strength", rule.get("strength")) in
        ("inviolable", "strong_preference", "aspirational", "weak")
        for i, rule in enumerate(case.gold.get("commitment_rules", [])))
    return {"contract_tsr": status, "contract_reasons": reasons,
            "contract_unverifiable": unknown,
            "contract_reference_eligible": reference_eligible,
            "workflow_protocol_reasons": outcome.protocol_reasons,
            "intent_budget_reported_correctly": report_fidelity,
            "budget_numeric_satisfied": budget_satisfied,
            "metric_protocol": "audit-v4-units", "execution_tsr_without_budget_echo": outcome.status}
