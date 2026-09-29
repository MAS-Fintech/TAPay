"""Shared test helpers: synthetic case builder + frozen dataset path."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from payment.cases import PaymentCase

CASES_PATH = Path(__file__).resolve().parent.parent / "datasets" / "master_311.jsonl"

MAIN = "Payment Main Supervisor"
EVAL = "Payment Evaluation Supervisor"
GEN = "Payment Action Generator"
REV = "Payment Revisor"
E1 = "Request/Intent Consistency Evaluator"
E2 = "Environment/Feasibility Evaluator"
E3 = "Transaction/Outcome Evaluator"


def make_environment(
    *,
    facts: Optional[Dict[str, Any]] = None,
    action_catalog: Optional[List[Dict[str, Any]]] = None,
    successor_state: Optional[Dict[str, Any]] = None,
    successor_visible_after_turn_id: Optional[int] = None,
) -> Dict[str, Any]:
    state = {
        "action_catalog": action_catalog
        or [
            {
                "id": "act-test",
                "operation": "swap",
                "parameters": {
                    "amount": {"minimum": 1e-06, "required": True, "type": "number"},
                    "asset": {"required": True, "type": "string"},
                    "slippage": {
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "required": True,
                        "type": "number",
                    },
                    "target_asset": {"required": True, "type": "string"},
                },
            }
        ],
        "asset_catalog": ["ETH", "USDT"],
        "facts": facts
        or {
            "approval_present": True,
            "balance": 100000.0,
            "max_amount": 100000.0,
            "max_slippage": 100.0,
            "plan_valid": True,
            "quote_valid": True,
            "remaining_budget": 100000.0,
            "remaining_time_s": 600.0,
            "risk_score": 0.1,
            "route_available": True,
        },
        "state_version": 1,
    }
    return {
        "conversation_start_at": "2026-08-30T08:38:08+08:00",
        "timezone": "Asia/Hong_Kong",
        "network": "ethereum",
        "schema_version": "payment-environment-v2",
        "initial_state": state,
        "successor_state": successor_state,
        "successor_visible_after_turn_id": successor_visible_after_turn_id,
    }


def make_case(
    *,
    case_id: str = "case-test",
    dataset_kind: str = "static",
    conversation: Optional[List[Dict[str, Any]]] = None,
    environment: Optional[Dict[str, Any]] = None,
    expected_behavior: str = "EXECUTE",
    contract: Optional[Dict[str, Any]] = None,
    allowed_action_ids: Optional[List[str]] = None,
    runtime_action_id_map: Optional[Dict[str, str]] = None,
    simulation_max_calls: Optional[int] = 2,
    simulation_per_proposal_max_calls: Optional[int] = 1,
    simulation_permitted: bool = True,
    legal_trajectories: Optional[List[List[str]]] = None,
    expected_handoff_edges: Optional[List[List[str]]] = None,
) -> PaymentCase:
    if contract is None:
        contract = {
            "amount": 100.0,
            "asset": "ETH",
            "operation": "swap",
            "slippage": 0.5,
            "target_asset": "USDT",
            "field_status": {
                "amount": "PRESENT_VALID",
                "asset": "PRESENT_VALID",
                "operation": "PRESENT_VALID",
                "slippage": "PRESENT_VALID",
                "target_asset": "PRESENT_VALID",
            },
        }
    task_reference = {
        "allowed_action_ids": allowed_action_ids or ["uniswap.execute_swap"],
        "allowed_operations": [contract.get("operation", "swap")],
        "requested_action_contract": contract,
        "runtime_action_id_map": runtime_action_id_map
        or {"uniswap.execute_swap": "act-test"},
        "simulation_max_calls": simulation_max_calls,
        "simulation_per_proposal_max_calls": simulation_per_proposal_max_calls,
        "simulation_permitted": simulation_permitted,
    }
    return PaymentCase(
        case_id=case_id,
        dataset_kind=dataset_kind,
        conversation=conversation
        or [{"role": "user", "turn_id": 1, "content": "swap 100 ETH to USDT"}],
        environment=environment or make_environment(),
        gold={
            "expected_behavior": expected_behavior,
            "canonical_intent": {},
            "expected_runtime": "SIMULATED_EXECUTED",
        },
        task_reference=task_reference,
        workflow_reference=(
            {
                "B000": {
                    "legal_trajectories": legal_trajectories or [],
                    "expected_handoff_edges": expected_handoff_edges or [],
                }
            }
            if legal_trajectories is not None
            or expected_handoff_edges is not None
            else None
        ),
        split="test_candidate",
    )


def expand_legal_to_raw(legal: List[str], layout) -> List[str]:
    """Inverse of trajectory normalization, mirroring graph mechanics.

    Used to test that normalize_trajectory(expand(legal)) == legal for every
    reviewed reference trajectory.
    """
    raw = [legal[0]]
    for a, b in zip(legal, legal[1:]):
        handled = False
        for sup, members in layout.supervisors.items():
            if a == sup and b in members:
                raw.append(b)  # dispatch
                handled = True
                break
            if a in members and b in members and a in layout.leaf_members:
                raw.extend([sup, b])  # collapsed intra-team hop
                handled = True
                break
            if a in members and b == sup:
                raw.append(b)  # member returns to its supervisor
                handled = True
                break
        if not handled:
            raw.append(b)  # team-boundary crossing (e.g. EvalSup -> Main)
    return raw
