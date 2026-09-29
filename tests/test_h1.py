"""H1 module tests: schema rules, structurer, versioning, B100 wiring."""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from multiagent.llm import ScriptedChatModel
from payment.h1.schema import (
    FieldReading,
    StructuredIntent,
    compute_diff,
    finalize_intent,
    Commitment,
)
from payment.h1.structurer import H1IntentStructurer, intent_injection_block
from payment.hooks import build_hooks
from payment.mock_replay import _last_embedded_output, _last_receipt_id
from payment.runner import run_case

from helpers import E1, E2, E3, EVAL, GEN, MAIN, REV, make_case, make_environment


# ------------------------------------------------------------------ schema

def _reading(value, status="explicit", hardness="hard"):
    return FieldReading(value=value, status=status, evidence="...", hardness=hardness)


def test_finalize_fail_closed_on_missing_required():
    intent = StructuredIntent(
        scheme="swap",
        fields={
            "operation": _reading("swap"),
            "amount": _reading(100.0),
            "asset": _reading("ETH"),
            "target_asset": _reading("USDT"),
            "slippage": _reading(None, status="missing", hardness="soft"),
        },
    )
    out = finalize_intent(intent)
    assert "slippage" in out.missing_required
    assert out.requires_clarification is True
    assert "slippage" in (out.clarification_question or "")


def test_finalize_ok_when_all_required_present():
    intent = StructuredIntent(
        scheme="swap",
        fields={
            "operation": _reading("swap"),
            "amount": _reading(100.0),
            "asset": _reading("ETH"),
            "target_asset": _reading("USDT"),
            "slippage": _reading(0.5),
        },
    )
    out = finalize_intent(intent)
    assert out.missing_required == []
    assert out.requires_clarification is False


def test_finalize_uses_live_catalog_required_fields():
    intent = StructuredIntent(
        scheme="deposit",
        fields={"operation": _reading("deposit"), "amount": _reading(5.0),
                "asset": _reading(None, status="missing")},
    )
    out = finalize_intent(intent, catalog_required={"deposit": ["amount", "asset"]})
    assert "asset" in out.missing_required
    assert out.requires_clarification is True


def test_compute_diff_tracks_status_and_value():
    v1 = StructuredIntent(version=1, scheme="swap",
                          fields={"asset": _reading("USDC"),
                                  "amount": _reading(100.0)})
    v2 = StructuredIntent(version=1, scheme="swap",
                          fields={"asset": _reading("DAI"),
                                  "amount": _reading(100.0)})
    diff = compute_diff(v1, v2)
    assert list(diff) == ["asset"]
    assert diff["asset"]["to"] == ("explicit", "DAI")


def test_finalize_versions_and_diffs():
    v1 = finalize_intent(StructuredIntent(
        scheme="borrow",
        fields={"operation": _reading("borrow"), "amount": _reading(15000.0),
                    "asset": _reading("USDC")}))
    v2 = finalize_intent(StructuredIntent(
        scheme="borrow",
        fields={"operation": _reading("borrow"), "amount": _reading(15000.0),
                "asset": _reading("DAI")}), previous=v1)
    assert v2.version == 2 and v2.supersedes == 1
    assert v2.diff_from_previous["asset"]["to"] == ("explicit", "DAI")
    # a user-initiated revision must NOT force clarification
    assert v2.requires_clarification is False


def test_finalize_unresolved_conflict_forces_clarification():
    intent = StructuredIntent(
        scheme="swap",
        fields={"operation": _reading("swap"), "amount": _reading(1.0),
                "asset": _reading("ETH"), "target_asset": _reading("USDT"),
                "slippage": _reading(0.01)},
        conflicts=["user stated both 'exactly 100' and 'at most 50' with no resolution"],
    )
    out = finalize_intent(intent)
    assert out.requires_clarification is True


def test_constraint_never_dual_listed_hard_wins():
    """The exact wording from the observed B111 failure: same requirement in
    both lists with different surface phrasing must be deduped, hard side kept."""
    intent = StructuredIntent(
        scheme="swap",
        fields={"operation": _reading("swap"), "amount": _reading(8500.0),
                "asset": _reading("DAI"), "target_asset": _reading("ETH"),
                "slippage": _reading(0.01)},
        hard_constraints=["Slippage must not exceed 0.01%.",
                          "Receive at least 2.5 ETH."],
        soft_preferences=["Use Uniswap.",
                          "The user hopes to receive at least 2.5 ETH."],
    )
    out = finalize_intent(intent)
    assert "Receive at least 2.5 ETH." in out.hard_constraints
    assert out.soft_preferences == ["Use Uniswap."]


def test_graded_commitments_derive_lists():
    """Graded commitments are the primary representation; the plain hard/soft
    lists are derived from them (inviolable -> hard, everything else -> soft)."""
    from payment.h1.schema import Commitment

    intent = StructuredIntent(
        scheme="swap",
        fields={"operation": _reading("swap"), "amount": _reading(8500.0),
                "asset": _reading("DAI"), "target_asset": _reading("ETH"),
                "slippage": _reading(0.01)},
        commitments=[
            Commitment(rule="Slippage must not exceed 0.01%.",
                       level="inviolable", value=0.01, unit="percent",
                       direction="max", evidence="doesn't exceed 0.0100%"),
            Commitment(rule="Receive at least 2.5 ETH.",
                       level="aspirational_target", value=2.5, unit="ETH",
                       direction="min", evidence="hoping to receive ~2.5 ETH"),
            Commitment(rule="Use Uniswap.", level="weak_preference"),
        ],
    )
    out = finalize_intent(intent)
    assert out.hard_constraints == ["Slippage must not exceed 0.01%."]
    assert "Receive at least 2.5 ETH." in out.soft_preferences
    assert "Use Uniswap." in out.soft_preferences
    # the aspirational target must not leak into the hard side
    assert not any("2.5 ETH" in h for h in out.hard_constraints)
    # unit/direction survive
    c = out.commitments[0]
    assert c.unit == "percent" and c.direction == "max" and c.value == 0.01


# ------------------------------------------------------------------ structurer

_INTENT_V1 = {
    "scheme": "borrow",
    "fields": {
        "operation": {"value": "borrow", "status": "explicit", "evidence": "borrow 15000 USDC", "hardness": "hard"},
        "amount": {"value": 15000.0, "status": "explicit", "evidence": "15000", "hardness": "hard"},
        "asset": {"value": "USDC", "status": "explicit", "evidence": "USDC", "hardness": "hard"},
    },
    "hard_constraints": ["asset = USDC"],
    "soft_preferences": [],
    "conflicts": [],
    "requires_clarification": False,
    "clarification_question": None,
}

_INTENT_V2 = {
    **_INTENT_V1,
    "fields": {**_INTENT_V1["fields"],
               "asset": {"value": "DAI", "status": "explicit",
                         "evidence": "use DAI instead", "hardness": "hard"}},
    "hard_constraints": ["asset = DAI"],
}


def _structurer_model():
    def responder(role, idx, messages, mode):
        assert role == "Intent Structurer"
        return dict(_INTENT_V1 if idx == 0 else _INTENT_V2)

    return ScriptedChatModel(responder=responder)


def test_structurer_prompt_carries_environment_snapshot():
    structurer = H1IntentStructurer(_structurer_model())
    structurer.configure(
        catalog_required={"borrow": ["amount", "asset"]},
        conversation_start_at="2026-08-30T08:00:00+08:00",
        timezone="Asia/Hong_Kong",
        environment_snapshot={"network": "ethereum", "facts": {"balance": 5.0},
                              "action_catalog": [{"operation": "borrow"}]},
    )
    messages = structurer._prompt("user: borrow 5 USDC", None)
    assert "ethereum" in messages[0].content
    assert '"balance": 5.0' in messages[0].content
    assert "2026-08-30T08:00:00+08:00" in messages[0].content


def test_structurer_structure_and_versioning():
    structurer = H1IntentStructurer(_structurer_model())
    structurer.configure(catalog_required={"borrow": ["amount", "asset"]},
                         conversation_start_at="2026-08-30T08:00:00+08:00",
                         timezone="Asia/Hong_Kong")
    v1 = structurer.structure("user: I'll borrow 15000 USDC.")
    assert v1.version == 1 and v1.scheme == "borrow"
    assert v1.requires_clarification is False
    v2 = structurer.on_user_event(["Actually, use DAI instead of USDC."],
                                  "user: borrow 15000 USDC\nuser: use DAI instead")
    assert v2 is not None
    assert v2.version == 2 and v2.supersedes == 1
    assert v2.field("asset").value == "DAI"
    assert "asset" in v2.diff_from_previous
    block = intent_injection_block(v2)
    assert "v2" in block and "supersedes v1" in block


# ------------------------------------------------------------------ B100 wiring

def _team_responder(captured):
    def responder(role, idx, messages, mode):
        captured.setdefault(role, []).extend(
            str(m.content) for m in messages if hasattr(m, "content")
        )
        if mode == "structured":
            if role in (MAIN, EVAL):
                main_queue = [GEN, EVAL, REV, EVAL, "FINISH"]
                eval_queue = [E1, E2, E3, MAIN, MAIN]
                queue = main_queue if role == MAIN else eval_queue
                return {"thoughts": "t", "next": queue[min(idx, len(queue) - 1)],
                        "messages": "m", "intermediate_output": "{{}}",
                        "background": "b"}
            if role == GEN:
                return {"intermediate_output": json.dumps({
                    "proposal": {"operation": "borrow", "action_id": "act-test",
                                 "parameters": {"amount": 15000.0, "asset": "DAI"},
                                 "receipt_id": _last_receipt_id(messages)},
                    "decision_recommendation": "EXECUTE", "rationale": "ok"})}
            if role in (E1, E2, E3):
                out = _last_embedded_output(messages)
                evaluation = dict(out.get("evaluation", {}))
                evaluation[role] = {"verdict": "PASS", "score": 10, "reason": "ok"}
                out["evaluation"] = evaluation
                return {"intermediate_output": json.dumps(out)}
            if role == REV:
                out = _last_embedded_output(messages)
                proposal = out.get("proposal", {})
                out["final_decision"] = {
                    "final_behavior": "EXECUTE",
                    "action_id": proposal.get("action_id"),
                    "operation": "borrow",
                    "parameters": proposal.get("parameters"),
                    "receipt_id": proposal.get("receipt_id"),
                    "rationale": "final",
                }
                return {"intermediate_output": json.dumps(out)}
            if role == "Intent Structurer":
                return dict(_INTENT_V1 if idx == 0 else _INTENT_V2)
            raise ValueError(role)
        # generate mode
        if role == GEN:
            step = ["env", "sim", "final"][min(idx, 2)]
            if step == "env":
                return AIMessage(content="", tool_calls=[
                    {"name": "environment_query", "args": {"scope": "current"},
                     "id": f"g{idx}", "type": "tool_call"}])
            if step == "sim":
                return AIMessage(content="", tool_calls=[
                    {"name": "simulate_transaction",
                     "args": {"action_id": "act-test",
                              "parameters": json.dumps(
                                  {"amount": 15000.0, "asset": "DAI"})},
                     "id": f"g{idx}", "type": "tool_call"}])
            return AIMessage(content="generator report")
        if role in (E2, E3) and idx == 0:
            return AIMessage(content="", tool_calls=[
                {"name": "environment_query", "args": {"scope": "current"},
                 "id": f"{role}{idx}", "type": "tool_call"}])
        return AIMessage(content=f"{role} report")

    return responder


def _dynamic_revision_case():
    env = make_environment(
        successor_state={
            "action_catalog": make_environment()["initial_state"]["action_catalog"],
            "asset_catalog": ["USDC", "DAI"],
            "facts": {"approval_present": True, "balance": 50000.0,
                      "max_amount": 50000.0, "quote_valid": True,
                      "plan_valid": True, "route_available": True,
                      "risk_score": 0.1, "max_slippage": 100.0,
                      "remaining_budget": 50000.0, "remaining_time_s": 300.0},
            "state_version": 2,
        },
        successor_visible_after_turn_id=2,
    )
    contract = {
        "amount": 15000.0, "asset": "DAI", "operation": "borrow",
        "field_status": {"amount": "PRESENT_VALID", "asset": "PRESENT_VALID",
                         "operation": "PRESENT_VALID",
                         "slippage": "NOT_APPLICABLE",
                         "target_asset": "NOT_APPLICABLE"},
    }
    return make_case(
        dataset_kind="dynamic",
        conversation=[
            {"role": "user", "turn_id": 1,
             "content": "I'll borrow 15000 USDC before the day ends."},
            {"role": "user", "turn_id": 2,
             "content": "Actually, use DAI instead of USDC for the same borrow."},
        ],
        environment=env,
        expected_behavior="EXECUTE",
        contract=contract,
        allowed_action_ids=["aave.borrow"],
        runtime_action_id_map={"aave.borrow": "act-test"},
    )


def test_b100_intent_reaches_team_and_versions_on_user_event():
    case = _dynamic_revision_case()
    captured: dict = {}
    # one shared mock for the team and the structurer
    model = ScriptedChatModel(responder=_team_responder(captured))
    hooks = build_hooks(h1=True, llm=model)
    result = run_case(case, model, hooks)

    assert result.prediction.failure is None
    assert result.prediction.final_behavior == "EXECUTE"
    # structurer produced v1 at start and v2 after the user event release
    structurer_calls = model.call_counts.get("Intent Structurer|structured", 0)
    assert structurer_calls == 2
    # v1 visible to the generator; v2 visible to later members (evaluators)
    gen_text = "\n".join(captured.get(GEN, []))
    assert "Current structured intent (v1)" in gen_text
    later_text = "\n".join(
        m for role in (EVAL, E2, E3) for m in captured.get(role, [])
    )
    assert "Current structured intent (v2" in later_text
    assert "DAI" in later_text


def test_b000_unchanged_without_h1():
    case = _dynamic_revision_case()
    captured: dict = {}
    model = ScriptedChatModel(responder=_team_responder(captured))
    hooks = build_hooks()
    result = run_case(case, model, hooks)
    assert result.prediction.failure is None
    assert "Intent Structurer|structured" not in model.call_counts
    gen_text = "\n".join(captured.get(GEN, []))
    assert "Current structured intent" not in gen_text


def test_structurer_context_refreshes_to_post_event_state():
    """The successor catalog adds DAI; the re-structurer must see it."""
    case = _dynamic_revision_case()
    assert "DAI" not in case.environment["initial_state"]["asset_catalog"]
    assert "DAI" in case.environment["successor_state"]["asset_catalog"]

    captured: dict = {}
    model = ScriptedChatModel(responder=_team_responder(captured))
    structurer = H1IntentStructurer(model)
    snapshots_at_restructure: list = []
    original = structurer.on_user_event

    def spy_on_user_event(events, conversation):
        snapshots_at_restructure.append(
            list(structurer.environment_snapshot.get("asset_catalog", []))
        )
        return original(events, conversation)

    structurer.on_user_event = spy_on_user_event
    hooks = build_hooks()
    object.__setattr__(hooks, "h1", structurer)
    result = run_case(case, model, hooks)

    assert result.prediction.failure is None
    assert snapshots_at_restructure, "re-structuring must have fired"
    assert "DAI" in snapshots_at_restructure[-1]



def test_inviolable_slippage_commitment_never_unresolved():
    """audit-v2 regression: 'Do not exceed 2% slippage' was categorized as
    'other' (slippage had no category) and landed in unresolved_hard_constraints,
    forcing a bogus clarification on a perfectly executable request. Slippage
    is a catalog-expressible parameter — it must never be unresolved."""
    intent = StructuredIntent(
        scheme="swap",
        fields={
            "operation": _reading("swap"),
            "amount": _reading(2.998),
            "asset": _reading("WETH"),
            "target_asset": _reading("APU"),
            "slippage": _reading(2.0),
        },
        commitments=[Commitment(
            rule="Do not exceed 2% slippage",
            level="inviolable",
            category="other",          # the mis-assignment seen in the wild
            evidence="slippage hard cap 2%, can't go over",
            value=2.0, unit="percent", direction="max",
        )],
    )
    out = finalize_intent(intent)
    assert out.commitments[0].category == "slippage_cap"
    assert out.unresolved_hard_constraints == []
    assert out.requires_clarification is False


def test_finalize_conflict_backfills_question():
    """A conflict-driven RC must always carry an actionable question."""
    intent = StructuredIntent(
        scheme="swap",
        fields={"operation": _reading("swap"), "amount": _reading(1.0),
                "asset": _reading("ETH"), "target_asset": _reading("USDT"),
                "slippage": _reading(0.01)},
        conflicts=["user stated both 'exactly 100' and 'at most 50' with no resolution"],
    )
    out = finalize_intent(intent)
    assert out.requires_clarification is True
    assert out.clarification_question
    assert "exactly 100" in out.clarification_question
