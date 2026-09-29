"""H2 module tests: rule layer, aggregation, early stop, B010 wiring."""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from multiagent.llm import ScriptedChatModel
from payment.h2.verifier import H2AlignmentVerifier, evidence_block
from payment.hooks import build_hooks
from payment.mock_replay import _last_embedded_output, _last_receipt_id
from payment.runner import run_case

from helpers import E1, E2, E3, EVAL, GEN, MAIN, REV, make_case

_ENV = {
    "action_catalog": [
        {"id": "act-test", "operation": "swap",
         "parameters": {
             "amount": {"minimum": 1e-06, "required": True, "type": "number"},
             "asset": {"required": True, "type": "string"},
             "slippage": {"minimum": 0.0, "maximum": 1.0, "required": True,
                          "type": "number"},
             "target_asset": {"required": True, "type": "string"}}}
    ],
    "asset_catalog": ["ETH", "USDT"],
    "facts": {"approval_present": True, "balance": 100000.0, "max_amount": 100000.0,
              "max_slippage": 100.0, "quote_valid": True, "plan_valid": True,
              "route_available": True, "risk_score": 0.1},
    "state_version": 1,
}

_GOOD_PROPOSAL = {
    "proposal": {
        "operation": "swap", "action_id": "act-test",
        "parameters": {"amount": 100.0, "asset": "ETH", "target_asset": "USDT",
                       "slippage": 0.5},
        "receipt_id": "r1",
    },
    "decision_recommendation": "EXECUTE",
}

_RECEIPT = {
    "receipt_id": "r1", "action_id": "act-test", "operation": "swap",
    "parameters": {"amount": 100.0, "asset": "ETH", "target_asset": "USDT",
                   "slippage": 0.5},
    "state_version": 1, "status": "SIMULATED_EXECUTED", "failure_reasons": [],
}


def _verifier():
    model = ScriptedChatModel(
        responder=lambda role, idx, messages, mode: {
            "observations": [], "alignment_risk": "low", "reason": "ok",
        }
    )
    v = H2AlignmentVerifier(model)
    v.configure(visible_conversation="user: swap 100 ETH to USDT",
                simulation_max_calls=2)
    return v, model


def test_rule_layer_pass_and_semantic_runs():
    v, model = _verifier()
    out = v.verify(_GOOD_PROPOSAL, [_RECEIPT], _ENV)
    assert out["overall"] == "PASS"
    assert out["hard_gate"] is None
    assert out["semantic"]["observations"] == []
    assert model.call_counts.get("Alignment Verifier|structured") == 1


def test_rule_reject_skips_semantic():
    v, model = _verifier()
    bad = {
        "proposal": {"operation": "swap", "action_id": "act-nope",
                     "parameters": {"amount": 100.0}},
    }
    out = v.verify(bad, [], _ENV)
    assert out["overall"] == "REJECT"
    assert out["hard_gate"] == "REJECT"
    # early termination: the LLM semantic layer must not run
    assert model.call_counts.get("Alignment Verifier|structured", 0) == 0


def test_stale_receipt_rejected():
    v, _ = _verifier()
    stale = dict(_RECEIPT, state_version=0)
    out = v.verify(_GOOD_PROPOSAL, [stale], _ENV)
    assert out["overall"] == "REJECT"
    assert any(c["check"] == "receipt_binding" and c["verdict"] == "FAIL"
               for c in out["rule_checks"])


def test_parameter_schema_failure():
    v, _ = _verifier()
    no_slippage = {
        "proposal": {"operation": "swap", "action_id": "act-test",
                     "parameters": {"amount": 100.0, "asset": "ETH",
                                    "target_asset": "USDT"}},
    }
    out = v.verify(no_slippage, [_RECEIPT], _ENV)
    assert out["overall"] == "REJECT"
    assert any("slippage" in c["reason"] for c in out["rule_checks"])


def test_facts_gate_failure():
    v, _ = _verifier()
    env = {**_ENV, "facts": {**_ENV["facts"], "balance": 10.0,
                             "max_amount": 100000.0}}
    out = v.verify(_GOOD_PROPOSAL, [_RECEIPT], env)
    assert out["overall"] == "REJECT"
    assert any("balance" in c["reason"] for c in out["rule_checks"])


def test_no_proposal_path():
    v, _ = _verifier()
    out = v.verify({"proposal": {"operation": None, "action_id": None,
                                 "parameters": None},
                    "decision_recommendation": "REQUEST_CLARIFICATION"},
                   [], _ENV)
    assert out["overall"] == "NOT_APPLICABLE"
    assert out["hard_gate"] is None


def test_receipt_binding_only_applies_to_execute():
    v, _ = _verifier()
    # no receipt, but the recommendation is BLOCK -> receipt_binding must not
    # fire (nothing is owed), overall stays NOT_APPLICABLE-to-PASS range
    block_proposal = {
        "proposal": {"operation": "swap", "action_id": "act-test",
                     "parameters": {"amount": 100.0, "asset": "ETH",
                                    "target_asset": "USDT", "slippage": 0.5}},
        "decision_recommendation": "BLOCK",
    }
    out = v.verify(block_proposal, [], _ENV)
    assert out["hard_gate"] is None
    assert all(c["check"] != "receipt_binding" for c in out["rule_checks"])
    # EXECUTE without a bound receipt -> REJECT
    out2 = v.verify(_GOOD_PROPOSAL, [], _ENV)
    assert out2["hard_gate"] == "REJECT"
    assert any(c["check"] == "receipt_binding" and c["verdict"] == "FAIL"
               for c in out2["rule_checks"])


def test_semantic_observations_never_veto():
    v, model = _verifier()
    # make the semantic layer report a candidate mismatch
    model.responder = lambda role, idx, messages, mode: {
        "observations": ["amount wording differs: 'about 100' vs 100.0"],
        "alignment_risk": "medium", "reason": "candidate mismatch",
    }
    out = v.verify(_GOOD_PROPOSAL, [_RECEIPT], _ENV)
    assert out["overall"] == "PASS_WITH_OBSERVATIONS"
    assert out["hard_gate"] is None


def test_intent_field_alignment_gate():
    from payment.h1.schema import FieldReading, StructuredIntent

    intent = StructuredIntent(
        version=1,
        scheme="swap",
        fields={
            "operation": FieldReading(value="swap", status="explicit"),
            "amount": FieldReading(value=100.0, status="explicit"),
            "asset": FieldReading(value="ETH", status="explicit"),
            "target_asset": FieldReading(value="USDT", status="explicit"),
            "slippage": FieldReading(value=0.5, status="explicit"),
        },
    )
    v, model = _verifier()
    # aligned proposal passes the intent check
    out = v.verify(_GOOD_PROPOSAL, [_RECEIPT], _ENV, intent=intent)
    assert any(c["check"] == "intent_field_alignment" and c["verdict"] == "PASS"
               for c in out["rule_checks"])
    # a substituted amount is a hard FAIL
    bad = {
        "proposal": {"operation": "swap", "action_id": "act-test",
                     "parameters": {"amount": 200.0, "asset": "ETH",
                                    "target_asset": "USDT", "slippage": 0.5}},
        "decision_recommendation": "EXECUTE",
    }
    out = v.verify(bad, [_RECEIPT], _ENV, intent=intent)
    assert out["hard_gate"] == "REJECT"
    assert any("amount" in c["reason"] for c in out["rule_checks"])
    # filling a REQUIRED field the intent marks missing is a hard FAIL
    # (unfounded completion); optional absent fields stay free to fill.
    intent2 = intent.model_copy(deep=True)
    intent2.fields["slippage"] = FieldReading(value=None, status="missing")
    intent2.missing_required = ["slippage"]
    out2 = v.verify(_GOOD_PROPOSAL, [_RECEIPT], _ENV, intent=intent2)
    assert out2["hard_gate"] == "REJECT"
    assert any("slippage" in c["reason"] for c in out2["rule_checks"])


def test_evidence_block_veto_contract():
    v, _ = _verifier()
    out = v.verify({"proposal": {"operation": "swap", "action_id": "act-nope",
                                 "parameters": {}}}, [], _ENV)
    block = evidence_block(out)
    assert "VETO CONTRACT" in block
    assert evidence_block(None) == ""


# ------------------------------------------------------------------ B010 wiring

def _team_responder(captured):
    def responder(role, idx, messages, mode):
        captured.setdefault(role, []).extend(
            str(m.content) for m in messages if hasattr(m, "content")
        )
        if role == "Alignment Verifier":
            return {"observations": [], "alignment_risk": "low",
                    "reason": "ok"}
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
                    "proposal": {"operation": "swap", "action_id": "act-test",
                                 "parameters": {"amount": 100.0, "asset": "ETH",
                                                "target_asset": "USDT",
                                                "slippage": 0.5},
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
                    "operation": "swap",
                    "parameters": proposal.get("parameters"),
                    "receipt_id": proposal.get("receipt_id"),
                    "rationale": "final",
                }
                return {"intermediate_output": json.dumps(out)}
            raise ValueError(role)
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
                                  {"amount": 100.0, "asset": "ETH",
                                   "target_asset": "USDT", "slippage": 0.5})},
                     "id": f"g{idx}", "type": "tool_call"}])
            return AIMessage(content="generator report")
        if role in (E2, E3) and idx == 0:
            return AIMessage(content="", tool_calls=[
                {"name": "environment_query", "args": {"scope": "current"},
                 "id": f"{role}{idx}", "type": "tool_call"}])
        return AIMessage(content=f"{role} report")

    return responder


def test_b010_evidence_reaches_team_and_is_persisted():
    case = make_case()
    captured: dict = {}
    model = ScriptedChatModel(responder=_team_responder(captured))
    hooks = build_hooks(h2=True, llm=model)
    result = run_case(case, model, hooks)

    assert result.prediction.failure is None
    assert result.prediction.final_behavior == "EXECUTE"
    # CP1 + CP2: evidence produced at least twice (generator + eval return)
    assert len(result.h2_evidence_history) >= 2
    assert all(e["overall"] == "PASS" for e in result.h2_evidence_history)
    # the evidence rode the B channel to the evaluation team
    eval_text = "\n".join(
        m for role in (EVAL, E1, E2, E3) for m in captured.get(role, [])
    )
    assert "Independent alignment verification (H2)" in eval_text


def test_b000_produces_no_evidence():
    case = make_case()
    model = ScriptedChatModel(responder=_team_responder({}))
    hooks = build_hooks()
    result = run_case(case, model, hooks)
    assert result.h2_evidence_history == []
    assert "Alignment Verifier|structured" not in model.call_counts
