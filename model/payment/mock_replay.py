"""Offline replay driver — VALIDATION ONLY.

Builds a ScriptedChatModel that plays a *perfect* agent for a given case:
supervisors follow the case's reviewed legal trajectory, members emit the
contract-correct proposal and consistent evaluations.  It exists to validate
the pipeline end-to-end (instrumentation, runner, scorer) without an LLM
provider: a correct pipeline must score ~1.0 on every metric under replay.

This module is never imported by the production runner path; it reads gold /
reference fields that are invisible to the model at run time.
"""
from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from multiagent.llm import ScriptedChatModel
from multiagent.trajectory import TeamLayout
from payment.cases import PaymentCase
from payment.prompts import (
    E1,
    E2,
    E3,
    EVAL_MEMBERS,
    EVAL_SUPERVISOR,
    GENERATOR,
    MAIN_SUPERVISOR,
    REVISOR,
    team_layout,
)

_CATALOG_FIELDS = ("amount", "asset", "target_asset", "slippage")
_PURCHASE_FIELDS = ("sku", "quantity", "payment_method")


def successor_chain_length(case: PaymentCase) -> int:
    """Number of pending successor-chain entries (v4: a list of deltas; v3: a
    single full-state dict).  Each tool contact advances exactly one chain
    entry, so draining a chain of length L takes L contacts after the first
    query."""
    succ = case.environment.get("successor_state")
    if not succ:
        return 0
    return len(succ) if isinstance(succ, list) else 1


def derive_contract_action(case: PaymentCase):
    """VALIDATION-ONLY: (action_id, operation, params, timing_fields) derived
    from the case's reviewed task contract.  Shared by the per-baseline replay
    models; reads gold/reference fields invisible to the model at run time."""
    contract = case.task_reference.get("requested_action_contract", {})
    field_status = contract.get("field_status", {})
    action_map = case.task_reference.get("runtime_action_id_map") or {}
    # pick the runtime action whose catalog operation matches the contract
    # (multi-action catalogs list several ids; the first need not match).
    # v4 keeps the catalog at the environment top level; v3 nests it in
    # initial_state.
    catalog_ops = {
        a.get("id"): a.get("operation")
        for a in (
            case.environment["initial_state"].get("action_catalog")
            or case.environment.get("action_catalog", [])
        )
    }
    action_id = next(
        (rid for rid in action_map.values() if catalog_ops.get(rid) == contract.get("operation")),
        next(iter(action_map.values()), None),
    )
    operation = contract.get("operation")
    catalog_fields = (_PURCHASE_FIELDS if operation == "purchase"
                      else _CATALOG_FIELDS)
    params = {
        f: contract[f]
        for f in catalog_fields
        if field_status.get(f) == "PRESENT_VALID" and contract.get(f) is not None
    }
    # Intent-level timing fields live at the proposal/decision level.
    timing_fields = {
        f: contract[f]
        for f in ("time_window_seconds", "deadline_at")
        if field_status.get(f) == "PRESENT_VALID" and contract.get(f) is not None
    }
    return action_id, contract.get("operation"), params, timing_fields


def derive_supervisor_queues(legal_trajectory: List[str], layout: Optional[TeamLayout] = None):
    """Expand a normalized legal trajectory into per-supervisor routing queues.

    Every supervisor entry in the trajectory is one invocation deciding the
    next node; every collapsed intra-team hop (leaf member -> member of the
    same team) implies one extra invocation of that team's supervisor.
    """
    layout = layout or team_layout()
    queues: Dict[str, List[str]] = {sup: [] for sup in layout.supervisors}
    for i, node in enumerate(legal_trajectory):
        if i > 0:
            prev = legal_trajectory[i - 1]
            for sup, members in layout.supervisors.items():
                if (
                    prev != sup
                    and node != sup
                    and prev in members
                    and node in members
                    and prev in layout.leaf_members
                ):
                    queues[sup].append(node)
        if node in layout.supervisors:
            nxt = legal_trajectory[i + 1] if i + 1 < len(legal_trajectory) else None
            if node == MAIN_SUPERVISOR:
                queues[node].append(nxt if nxt else "FINISH")
            else:
                queues[node].append(nxt if nxt else MAIN_SUPERVISOR)
    return queues


def _parse_dict_prefix(text: str) -> Optional[Dict[str, Any]]:
    """Parse a leading ``{...}`` (balanced braces) from text, tolerating
    trailing content; try JSON first, then Python literal syntax."""
    text = text.strip()
    if not text.startswith("{"):
        return None
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[: i + 1]
                for parser in (json.loads, ast.literal_eval):
                    try:
                        obj = parser(candidate)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        return obj
                return None
    return None


def _last_embedded_output(messages: List[BaseMessage]) -> Dict[str, Any]:
    """Parse the most recent ``Final Output: {...}`` payload from history."""
    best: Optional[Dict[str, Any]] = None
    for msg in messages:
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            continue
        marker = "Final Output:"
        start = 0
        while True:
            pos = content.find(marker, start)
            if pos == -1:
                break
            obj = _parse_dict_prefix(content[pos + len(marker):])
            if obj is not None:
                best = obj
            start = pos + len(marker)
    return best or {}


def _last_receipt_id(messages: List[BaseMessage]) -> Optional[str]:
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == "simulate_transaction":
            try:
                payload = json.loads(msg.content)
            except Exception:
                continue
            receipt = payload.get("receipt")
            if isinstance(receipt, dict) and receipt.get("status") == "SIMULATED_EXECUTED":
                return receipt.get("receipt_id")
    return None


def _intent_from_gold(case: PaymentCase) -> Dict[str, Any]:
    """VALIDATION-ONLY: derive a plausible H1 intent draft from the case's
    canonical intent + field status (used to drive the scripted structurer in
    replay tests — it verifies H1 wiring, not model capability)."""
    contract = case.task_reference.get("requested_action_contract", {})
    field_status = contract.get("field_status", {})
    canonical = case.gold.get("canonical_intent", {})
    status_map = {
        "PRESENT_VALID": "explicit",
        "MISSING_REQUIRED": "missing",
        "UNSPECIFIED_OPTIONAL": "missing",
        "NOT_APPLICABLE": "unsupported",
    }
    fields: Dict[str, Any] = {}
    # The field set follows the case's own contract field_status (v4 DeFi
    # cases carry the seven canonical fields; v5 shopping contracts carry
    # sku/quantity/payment_method/max_total_usd), unioned with the canonical
    # seven so older contracts keep their shape.
    field_names = list(dict.fromkeys(
        [
            "operation", "amount", "asset", "target_asset", "slippage",
            "time_window_seconds", "deadline_at",
        ]
        + list(field_status.keys())
    ))
    for name in field_names:
        st = status_map.get(field_status.get(name, ""), "missing")
        value = canonical.get(name)
        if value is None:
            value = contract.get(name)
        fields[name] = {
            "value": value if st == "explicit" else None,
            "status": st,
            "evidence": "canonical intent (replay)",
            "hardness": "hard" if st == "explicit" else "soft",
        }
    return {
        "scheme": canonical.get("operation") or "unknown",
        "open_or_ambiguous": False,
        "fields": fields,
        "hard_constraints": [],
        "soft_preferences": [],
        "conflicts": [],
        "requires_clarification": False,
        "requires_block": False,
        "clarification_question": None,
    }


def build_replay_model(case: PaymentCase, h1_intent: Any = None) -> ScriptedChatModel:
    """Build the scripted replay model.  ``h1_intent`` is only used when the
    runner wires the H1 structurer: the mock then answers the 'Intent
    Structurer' role with this draft (``"auto"`` derives it from the case's
    canonical intent — validation only)."""
    wf = (case.workflow_reference or {}).get("B000") or {}
    legal = wf["legal_trajectories"][0]
    queues = derive_supervisor_queues(legal)

    contract = case.task_reference.get("requested_action_contract", {})
    field_status = contract.get("field_status", {})
    behavior = case.gold.get("expected_behavior")
    action_map = case.task_reference.get("runtime_action_id_map") or {}
    # pick the runtime action whose catalog operation matches the contract
    # (multi-action catalogs list several ids; the first need not match).
    # v4 keeps the catalog at the environment top level; v3 nests it in
    # initial_state.
    catalog_ops = {
        a.get("id"): a.get("operation")
        for a in (
            case.environment["initial_state"].get("action_catalog")
            or case.environment.get("action_catalog", [])
        )
    }
    action_id = next(
        (rid for rid in action_map.values() if catalog_ops.get(rid) == contract.get("operation")),
        next(iter(action_map.values()), None),
    )
    operation = contract.get("operation")
    catalog_fields = (_PURCHASE_FIELDS if operation == "purchase"
                      else _CATALOG_FIELDS)
    params = {
        f: contract[f]
        for f in catalog_fields
        if field_status.get(f) == "PRESENT_VALID" and contract.get(f) is not None
    }

    is_dynamic = case.dataset_kind == "dynamic"
    # Tool contacts needed to drain the successor chain BEFORE simulating:
    # each contact advances exactly one chain entry, so a chain of length L
    # needs L advancing contacts after the first query.
    chain_len = (len(case.environment.get("successor_state") or [])
                 if is_dynamic else 0)

    if h1_intent == "auto":
        h1_intent = _intent_from_gold(case)

    # Intent-level timing/budget fields live at the proposal/decision level.
    intent_fields = {
        f: contract[f]
        for f in ("time_window_seconds", "deadline_at", "max_total_usd")
        if field_status.get(f) == "PRESENT_VALID" and contract.get(f) is not None
    }

    # -- member step scripts -------------------------------------------------
    def _chain_draining_queries(sku: Optional[str] = None):
        """environment_query plus one extra contact per successor-chain entry
        (shopping cases use shop_get_product_details for the first extra
        contact, which is also the fresh-price re-check)."""
        steps = [("tool", "environment_query", {"scope": "current"})]
        extra = chain_len
        if sku is not None and extra:
            steps.append(("tool", "shop_get_product_details", {"sku": sku}))
            extra -= 1
        for _ in range(extra):
            steps.append(("tool", "environment_query", {"scope": "current"}))
        return steps

    def generator_steps():
        if behavior == "EXECUTE":
            steps = _chain_draining_queries(
                sku=params.get("sku") if operation == "purchase" else None)
            steps.append(("tool", "simulate_transaction",
                          {"action_id": action_id, "parameters": json.dumps(params)}))
            steps.append(("final",))
            return steps
        if behavior == "BLOCK":
            steps = _chain_draining_queries(
                sku=params.get("sku") if operation == "purchase" else None)
            steps.append(("final",))
            return steps
        return [("final",)]  # REQUEST_CLARIFICATION: nothing to simulate

    member_steps: Dict[str, List] = {
        GENERATOR: generator_steps(),
        E1: [("final",)],
        E2: [("tool", "environment_query", {"scope": "current"}), ("final",)],
        E3: [("tool", "environment_query", {"scope": "current"}), ("final",)],
        REVISOR: [("final",)],
    }

    def evaluator_verdict(role: str) -> Dict[str, Any]:
        if role == E1:
            if behavior == "REQUEST_CLARIFICATION":
                return {"verdict": "FAIL", "score": 2,
                        "reason": "required request fields are missing or ambiguous; clarification needed"}
            return {"verdict": "PASS", "score": 10,
                    "reason": "proposal matches the user's stated request"}
        if role == E2:
            if behavior == "BLOCK":
                return {"verdict": "FAIL", "score": 1,
                        "reason": "environment state forbids execution (risk/approval/balance)"}
            if behavior == "REQUEST_CLARIFICATION":
                return {"verdict": "N/A", "score": 5,
                        "reason": "no executable proposal to check"}
            return {"verdict": "PASS", "score": 10,
                    "reason": "proposal is feasible under the current environment state"}
        if role == E3:
            if behavior == "EXECUTE":
                return {"verdict": "PASS", "score": 10,
                        "reason": "successful receipt bound to the current state matches the proposal"}
            return {"verdict": "N/A", "score": 5,
                    "reason": "no transaction to check"}
        raise ValueError(role)

    # -- responder ------------------------------------------------------------
    def responder(role: str, idx: int, messages: List[BaseMessage], mode: str):
        if role == "Intent Structurer":
            if h1_intent is None:
                raise ValueError("replay model built without h1_intent")
            return dict(h1_intent)
        if role == "Alignment Verifier":
            return {"observations": [], "alignment_risk": "low",
                    "reason": "replay verdict"}
        if mode == "structured":
            if role in (MAIN_SUPERVISOR, EVAL_SUPERVISOR):
                # A final validation/terminal-event barrier can ask for another
                # decision without adding a workflow node. Replay the terminal
                # routing response; no production runner reads this oracle.
                nxt = queues[role][min(idx, len(queues[role]) - 1)]
                return {
                    "thoughts": f"replay step {idx}: route to {nxt}",
                    "next": nxt,
                    "messages": f"replay instruction for {nxt}",
                    "intermediate_output": "{{}}",
                    "background": "replay background",
                }
            base = _last_embedded_output(messages)
            if role == GENERATOR:
                receipt_id = _last_receipt_id(messages)
                proposal = {
                    "operation": operation,
                    "action_id": action_id if behavior == "EXECUTE" else None,
                    "parameters": dict(params) if behavior == "EXECUTE" else None,
                    "receipt_id": receipt_id,
                    **(
                        {k: v for k, v in intent_fields.items()}
                        if behavior == "EXECUTE"
                        else {}
                    ),
                }
                return {
                    "intermediate_output": json.dumps(
                        {
                            "proposal": proposal,
                            "decision_recommendation": behavior,
                            "rationale": "replay proposal",
                        }
                    )
                }
            if role in EVAL_MEMBERS:
                evaluation = dict(base.get("evaluation", {}))
                evaluation[role] = evaluator_verdict(role)
                out = dict(base)
                out["evaluation"] = evaluation
                return {"intermediate_output": json.dumps(out)}
            if role == REVISOR:
                proposal = base.get("proposal", {})
                final_decision = {
                    "final_behavior": behavior,
                    "action_id": proposal.get("action_id"),
                    "operation": proposal.get("operation", operation),
                    "parameters": proposal.get("parameters"),
                    "receipt_id": proposal.get("receipt_id"),
                    "rationale": "replay final decision",
                }
                for k in ("time_window_seconds", "deadline_at", "max_total_usd"):
                    if proposal.get(k) is not None:
                        final_decision[k] = proposal[k]
                out = dict(base)
                out["final_decision"] = final_decision
                return {"intermediate_output": json.dumps(out)}
            raise ValueError(f"unexpected structured role: {role}")

        # mode == "generate" (member ReAct steps)
        steps = member_steps[role]
        step = steps[min(idx, len(steps) - 1)]
        if step[0] == "tool":
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": step[1],
                        "args": step[2],
                        "id": f"call_{role}_{idx}",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content=f"{role} replay final report.")

    return ScriptedChatModel(responder=responder)
