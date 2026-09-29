"""H3 module tests: dynamic boundary policy, event directives, wiring."""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from multiagent.llm import ScriptedChatModel
from payment.environment import EnvironmentLedger
from payment.h1.schema import FieldReading, StructuredIntent
from payment.h3.policy import H3DecisionPolicy, directive_block
from payment.hooks import build_hooks
from payment.mock_replay import _last_embedded_output, _last_receipt_id
from payment.runner import run_case

from helpers import E1, E2, E3, EVAL, GEN, MAIN, REV, make_case, make_environment

_EXEC_DECISION = {
    "final_behavior": "EXECUTE",
    "action_id": "act-test",
    "operation": "swap",
    "parameters": {"amount": 100.0, "asset": "ETH", "target_asset": "USDT",
                   "slippage": 0.5},
    "receipt_id": "r1",
    "rationale": "ok",
}

_FACTS = {"approval_present": True, "balance": 100000.0, "max_amount": 100000.0,
          "max_slippage": 100.0, "quote_valid": True, "plan_valid": True,
          "route_available": True, "risk_score": 0.1}


def _env_state(facts=None, version=1):
    state = make_environment()["initial_state"]
    state = dict(state, facts=facts or dict(_FACTS), state_version=version)
    return state


def _policy():
    p = H3DecisionPolicy()
    p.configure(visible_conversation="user: swap 100 ETH to USDT")
    return p


# ------------------------------------------------------------------ boundaries

def test_risk_threshold_forces_block():
    p = _policy()
    env = _env_state(facts={**_FACTS, "risk_score": 0.95})
    out = p.decide(dict(_EXEC_DECISION), env)
    assert out["final_behavior"] == "BLOCK"
    assert "risk_score" in p.last_adjustment_reason


def test_missing_approval_forces_block():
    p = _policy()
    env = _env_state(facts={**_FACTS, "approval_present": False})
    out = p.decide(dict(_EXEC_DECISION), env)
    assert out["final_behavior"] == "BLOCK"


def test_h2_veto_forces_block():
    p = _policy()
    out = p.decide(dict(_EXEC_DECISION), _env_state(),
                   evidence={"hard_gate": "REJECT"})
    assert out["final_behavior"] == "BLOCK"
    assert "H2" in p.last_adjustment_reason


def test_h1_missing_required_forces_clarification():
    p = _policy()
    intent = StructuredIntent(
        scheme="swap",
        fields={"slippage": FieldReading(value=None, status="missing")},
        missing_required=["slippage"],
        requires_clarification=True,
    )
    out = p.decide(dict(_EXEC_DECISION), _env_state(), intent=intent)
    assert out["final_behavior"] == "REQUEST_CLARIFICATION"


def test_never_upgrades_non_execute():
    p = _policy()
    for behavior in ("BLOCK", "REQUEST_CLARIFICATION"):
        d = dict(_EXEC_DECISION, final_behavior=behavior, receipt_id=None)
        out = p.decide(d, _env_state())
        assert out["final_behavior"] == behavior


def test_reverification_rescues_stale_receipt():
    case = make_case()
    ledger = EnvironmentLedger(case)
    ledger.simulate("act-test", _EXEC_DECISION["parameters"])  # valid receipt v1
    # simulate a state change: bump version so the old receipt is stale
    ledger.current_state = dict(ledger.current_state, state_version=2)
    decision = dict(_EXEC_DECISION, receipt_id=ledger.receipts[0]["receipt_id"])
    p = _policy()
    out = p.decide(decision, ledger.current_state, ledger=ledger)
    assert out["final_behavior"] == "EXECUTE"
    assert out["receipt_id"] == ledger.receipts[-1]["receipt_id"]
    assert ledger.receipts[-1]["state_version"] == 2
    assert ledger.receipts[-1]["source"] == "h3_policy"
    assert "re-verified" in p.last_adjustment_reason


def test_reverification_impossible_without_budget_blocks():
    case = make_case(simulation_max_calls=1, simulation_per_proposal_max_calls=1)
    ledger = EnvironmentLedger(case)
    ledger.simulate("act-test", _EXEC_DECISION["parameters"])  # budget exhausted
    ledger.current_state = dict(ledger.current_state, state_version=2)
    decision = dict(_EXEC_DECISION, receipt_id=ledger.receipts[0]["receipt_id"])
    p = _policy()
    out = p.decide(decision, ledger.current_state, ledger=ledger)
    assert out["final_behavior"] == "BLOCK"
    assert "re-verification" in p.last_adjustment_reason


def test_valid_execute_untouched():
    case = make_case()
    ledger = EnvironmentLedger(case)
    ledger.simulate("act-test", _EXEC_DECISION["parameters"])
    decision = dict(_EXEC_DECISION, receipt_id=ledger.receipts[0]["receipt_id"])
    p = _policy()
    out = p.decide(decision, ledger.current_state, ledger=ledger)
    assert out is decision  # identity, no adjustment


# ------------------------------------------------------------------ fast loop

def test_event_directive_on_risk_rise():
    p = _policy()
    d = p.on_environment_event(
        ["risk score moved above threshold"],
        {"risk_score": 0.1, "quote_valid": True},
        {"risk_score": 0.95, "quote_valid": True},
    )
    assert d and "FORBIDDEN" in d


def test_event_directive_on_quote_expiry():
    p = _policy()
    d = p.on_environment_event(
        ["quote expired"],
        {"risk_score": 0.1, "quote_valid": True},
        {"risk_score": 0.1, "quote_valid": False},
    )
    assert d and "re-simulate" in d
    assert directive_block([d]).startswith("H3 dynamic boundary directives")
    assert directive_block([d]).startswith("H3 dynamic boundary directives")


def test_event_directive_on_quote_expiry_reverify_policy():
    p = H3DecisionPolicy(
        boundary={"risk_enter_offset": 0.0, "quote_policy": "re-verify"})
    d = p.on_environment_event(
        ["quote expired"],
        {"risk_score": 0.1, "quote_valid": True},
        {"risk_score": 0.1, "quote_valid": False},
    )
    assert d and "re-simulate" in d


# ------------------------------------------------------------------ B001 wiring

def _team_responder():
    def responder(role, idx, messages, mode):
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
                                 "parameters": dict(_EXEC_DECISION["parameters"]),
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
                              "parameters": json.dumps(_EXEC_DECISION["parameters"])},
                     "id": f"g{idx}", "type": "tool_call"}])
            return AIMessage(content="generator report")
        if role in (E2, E3) and idx == 0:
            return AIMessage(content="", tool_calls=[
                {"name": "environment_query", "args": {"scope": "current"},
                 "id": f"{role}{idx}", "type": "tool_call"}])
        return AIMessage(content=f"{role} report")

    return responder


def test_b001_identity_when_everything_valid():
    case = make_case()
    model = ScriptedChatModel(responder=_team_responder())
    hooks = build_hooks(h3=True)
    result = run_case(case, model, hooks)
    assert result.prediction.failure is None
    assert result.prediction.final_behavior == "EXECUTE"
    assert result.h3_adjustment == {}  # no adjustment needed


def test_b001_blocks_on_risky_state():
    env = make_environment(facts={**_FACTS, "risk_score": 0.9})
    case = make_case(environment=env, expected_behavior="BLOCK")
    model = ScriptedChatModel(responder=_team_responder())
    hooks = build_hooks(h3=True)
    result = run_case(case, model, hooks)
    # team recommended EXECUTE but the dynamic boundary must downgrade it
    assert result.prediction.final_behavior == "BLOCK"
    assert result.h3_adjustment["before"] == "EXECUTE"
    assert result.h3_adjustment["after"] == "BLOCK"


def test_b000_has_no_policy():
    case = make_case()
    model = ScriptedChatModel(responder=_team_responder())
    result = run_case(case, model, build_hooks())
    assert result.h3_directives == [] and result.h3_adjustment == {}


# ------------------------------------------------------- decide ordering (v4.3)

def test_substantive_veto_not_whitewashed_by_reverification():
    # inviolable economic violation + a stale receipt: the policy must BLOCK
    # BEFORE running any top-up simulation — a fresh successful receipt must
    # never wash away a substantive H2 veto.
    case = make_case()
    ledger = EnvironmentLedger(case)
    ledger.simulate("act-test", _EXEC_DECISION["parameters"])  # old receipt v1
    ledger.current_state = dict(ledger.current_state, state_version=2)
    decision = dict(_EXEC_DECISION, receipt_id=ledger.receipts[0]["receipt_id"])
    evidence = {
        "hard_gate": "REJECT",
        "rule_checks": [
            {"check": "economic_consistency", "verdict": "FAIL",
             "reason": "estimated_output 90.0 < min_output commitment 95.0 "
                       "[inviolable]"},
            {"check": "receipt_binding", "verdict": "FAIL",
             "reason": "environment state has changed"},
        ],
    }
    p = _policy()
    before = len(ledger.receipts)
    out = p.decide(decision, ledger.current_state, evidence=evidence,
                   ledger=ledger)
    assert out["final_behavior"] == "BLOCK"
    assert out["receipt_id"] is None
    assert "min_output" in p.last_adjustment_reason
    assert len(ledger.receipts) == before  # no whitewash simulation ran


def test_binding_only_veto_repaired_by_reverification():
    # a pure receipt-binding veto IS repairable: with budget remaining the
    # top-up simulation runs and EXECUTE survives.
    case = make_case()
    ledger = EnvironmentLedger(case)
    ledger.simulate("act-test", _EXEC_DECISION["parameters"])  # old receipt v1
    ledger.current_state = dict(ledger.current_state, state_version=2)
    decision = dict(_EXEC_DECISION, receipt_id=ledger.receipts[0]["receipt_id"])
    evidence = {
        "hard_gate": "REJECT",
        "rule_checks": [
            {"check": "receipt_binding", "verdict": "FAIL",
             "reason": "environment state has changed (receipt v1 vs current "
                       "v2): re-query the environment and re-run simulation"},
        ],
    }
    p = _policy()
    out = p.decide(decision, ledger.current_state, evidence=evidence,
                   ledger=ledger)
    assert out["final_behavior"] == "EXECUTE"
    assert out["receipt_id"] == ledger.receipts[-1]["receipt_id"]
    assert ledger.receipts[-1]["state_version"] == 2
    assert "re-verified" in p.last_adjustment_reason


def test_forbidden_exit_allowed_on_settled_chain_end():
    """V-shaped spike-then-recovery with the chain exhausted at the safe
    state: the settled terminal state satisfies the hold, exit is allowed."""
    from payment.h3.policy import H3DecisionPolicy as DecisionPolicy
    p = DecisionPolicy()
    # enter FORBIDDEN at risk 0.65 (chain not yet exhausted)
    d = p.on_environment_event([], {"risk_score": 0.1}, {"risk_score": 0.65},
                               chain_exhausted=False)
    assert d is not None and "FORBIDDEN" in d
    assert p.zone == "FORBIDDEN"
    # recovery to 0.2 is the LAST transition (chain exhausted): hold is
    # satisfied by settlement -> exit allowed
    p.on_environment_event([], {"risk_score": 0.65}, {"risk_score": 0.2},
                           chain_exhausted=True)
    assert p.zone == "AUTO"


def test_forbidden_exit_blocked_during_oscillation():
    """Mid-chain (more events pending): a single recovery event does NOT
    satisfy the minimum hold — the zone stays FORBIDDEN."""
    from payment.h3.policy import H3DecisionPolicy as DecisionPolicy
    p = DecisionPolicy()
    p.on_environment_event([], {"risk_score": 0.1}, {"risk_score": 0.65},
                           chain_exhausted=False)
    assert p.zone == "FORBIDDEN"
    p.on_environment_event([], {"risk_score": 0.65}, {"risk_score": 0.2},
                           chain_exhausted=False)  # more events may come
    assert p.zone == "FORBIDDEN"
    # a second event in the safe state (still mid-chain) satisfies the hold
    p.on_environment_event([], {"risk_score": 0.2}, {"risk_score": 0.2},
                           chain_exhausted=False)
    # zone machine only transitions on zone change; still FORBIDDEN until
    # a facts update that re-evaluates the exit with held >= 2 — simulate
    # by nudging risk inside the safe band
    p.on_environment_event([], {"risk_score": 0.2}, {"risk_score": 0.3},
                           chain_exhausted=False)
    assert p.zone in ("AUTO", "OBSERVE")
