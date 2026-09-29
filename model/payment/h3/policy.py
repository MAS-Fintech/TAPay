"""H3 DecisionPolicy: deterministic dynamic decision boundaries.

Two intervention points (both driven by the runner):
  * ``on_environment_event`` — fast loop: when dynamic events release a new
    environment state, produce a boundary directive for the team (re-verify,
    or execution forbidden) injected via the B channel.  Directives fire ONLY
    on risk-zone transitions (plus quote-expiry re-verify notices); a state
    change that crosses no boundary stays silent;
  * ``decide`` — slow loop at legal FINISH: validate/adjust the team's final
    decision against the CURRENT environment facts, the H1 intent (when
    present) and the latest H2 evidence (when present).  EXECUTE requires a
    receipt bound to the current state version; when missing and simulation
    budget remains, the policy performs one re-verification simulation itself
    (the "wait / re-verify" action of M3, offline-adapted).

Risk zones (three-zone state machine with hysteresis and a minimum hold):
  * FORBIDDEN (risk >= risk_enter): EXECUTE is downgraded to BLOCK;
  * OBSERVE   (risk_exit <= risk < risk_enter): EXECUTE stays possible but
    must hold a receipt bound to the current state/intent version;
  * AUTO      (risk < risk_exit): no risk-driven restriction.
  Entering FORBIDDEN requires risk >= risk_enter; leaving it requires
  risk <= risk_exit AND at least ``min_hold_events`` environment events since
  entry (anti-oscillation).  Zone state lives on the policy instance and is
  reset by ``configure()``.

The policy never upgrades a non-EXECUTE decision to EXECUTE, and never
exceeds the user's hard authorization boundary.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from payment.h2.verifier import receipt_bound_to_current

ZONES = ("AUTO", "OBSERVE", "FORBIDDEN")


class H3DecisionPolicy:
    #: Optional sub-features (all OFF unless named in ``features``):
    #:   slow_loop        — boundary-pressure accumulator that escalates to a
    #:                      BLOCK-leaning directive once pressure crosses a
    #:                      configurable threshold;
    #:   feedback_replan  — on_simulation_failure hook: category-keyed
    #:                      replan directive + verbatim-retry rejection;
    #:   clarify_policy   — value-of-information question ordering with a
    #:                      hard clarification budget (then BLOCK, not RC).
    FEATURES = ("slow_loop", "feedback_replan", "clarify_policy")

    def __init__(
        self,
        risk_threshold: float = 0.5,
        *,
        risk_enter: float = 0.6,
        risk_exit: float = 0.4,
        min_hold_events: int = 2,
        boundary: Optional[Dict[str, Any]] = None,
        features: Optional[set] = None,
        pressure_threshold: int = 3,
        clarify_budget: int = 2,
    ):
        # Backward compatible: a legacy single threshold maps onto the
        # FORBIDDEN-entry threshold when risk_enter is left at its default.
        self.risk_threshold = risk_threshold
        if risk_enter == 0.6 and risk_threshold != 0.5:
            risk_enter = risk_threshold
        # Optional learned boundary artifact (data/h3_boundary.json): the
        # FORBIDDEN-entry threshold becomes the declared threshold plus the
        # learned offset, and the quote-expiry stance follows the learned
        # policy. Without an artifact the policy behaves exactly as before.
        self.boundary = boundary or None
        if self.boundary:
            # v3 artifacts may recommend "no_preemption" (offset None): keep
            # the declared threshold and never preempt. Older artifacts carry
            # a numeric offset.
            offset = self.boundary.get("risk_enter_offset")
            if offset is not None:
                risk_enter = risk_enter + float(offset)
            # quote_policy is opt-in: only an artifact that explicitly sets
            # it changes the stance; v3 artifacts omit it entirely.
            self.quote_policy = self.boundary.get("quote_policy", "re-verify")
        else:
            # No artifact: keep the pre-bandit behavior exactly (re-verify on
            # stale quote, no hard quote gate). "forbid" is artifact-only.
            self.quote_policy = "re-verify"
        self.risk_enter = risk_enter
        self.risk_exit = risk_exit
        self.min_hold_events = min_hold_events
        # Sub-features (default OFF; orthogonal to the boundary artifact).
        self.features = set(features or ())
        unknown = self.features - set(self.FEATURES)
        if unknown:
            raise ValueError(f"unknown H3 features: {sorted(unknown)}")
        self.pressure_threshold = int(pressure_threshold)
        self.clarify_budget = int(clarify_budget)
        self.conversation = ""
        self.last_adjustment_reason: str = ""
        self.directives: List[str] = []
        self.zone = "AUTO"
        self.event_count = 0
        self.entered_at_event_count = 0
        # RQ3 instrumentation: per-event zone-machine audit trail.  Each
        # entry records one environment-event evaluation: the risk value,
        # the zone before/after, and whether the machine transitioned, was
        # held by hysteresis / minimum-hold, or stayed unchanged.
        self.zone_log: List[Dict[str, Any]] = []
        # slow_loop state
        self.pressure = 0
        self.escalation_active = False
        # feedback_replan state
        self._seen_failure_receipts: List[str] = []
        self._rejected_proposals: List[Dict[str, Any]] = []
        # clarify_policy state
        self._clarify_remaining = self.clarify_budget

    @classmethod
    def from_boundary_file(cls, path: str, **kwargs) -> "H3DecisionPolicy":
        """Build a policy whose boundary comes from a learned artifact JSON."""
        import json

        with open(path, encoding="utf-8") as f:
            boundary = json.load(f)
        return cls(boundary=boundary, **kwargs)

    def configure(self, **case_context) -> None:
        self.conversation = case_context.get("visible_conversation", "")
        self.directives = []
        self.last_adjustment_reason = ""
        self.zone = "AUTO"
        self.event_count = 0
        self.entered_at_event_count = 0
        self.zone_log = []
        self.pressure = 0
        self.escalation_active = False
        self._seen_failure_receipts = []
        self._rejected_proposals = []
        self._clarify_remaining = self.clarify_budget

    # ---------------------------------------------------------- slow_loop
    def _pressure_event(self, delta: int) -> Optional[str]:
        """Accumulate boundary pressure (slow_loop feature); emit the
        escalation directive once the threshold is crossed, then reset."""
        if "slow_loop" not in self.features:
            return None
        self.pressure = max(0, self.pressure + delta)
        if self.pressure >= self.pressure_threshold:
            self.pressure = 0
            self.escalation_active = True
            return (
                "boundary pressure has been accumulating over recent events: "
                "under the current evidence environment execution is NOT "
                "permitted — the correct terminal is BLOCK (or an explicit "
                "request for human confirmation), not EXECUTE."
            )
        return None

    # ---------------------------------------------------------- risk zones
    @staticmethod
    def _zone_of(risk: Optional[float], enter: float, exit_: float) -> Optional[str]:
        if risk is None:
            return None
        if risk >= enter:
            return "FORBIDDEN"
        if risk >= exit_:
            return "OBSERVE"
        return "AUTO"

    def _update_zone(self, current_facts: Dict[str, Any],
                     chain_exhausted: bool = False) -> Optional[str]:
        """Advance the zone state machine one environment event; return the
        transition "OLD->NEW" when the zone changed, else None."""
        self.event_count += 1
        risk = (current_facts or {}).get("risk_score")
        target = self._zone_of(risk, self.risk_enter, self.risk_exit)
        old = self.zone
        entry: Dict[str, Any] = {
            "event_count": self.event_count,
            "risk_score": risk,
            "zone_before": old,
            "target_zone": target,
            "chain_exhausted": chain_exhausted,
        }
        if target is None or target == old:
            self.zone_log.append({**entry, "decision": "unchanged"})
            return None
        if old == "FORBIDDEN" and target != "FORBIDDEN":
            # hysteresis + minimum hold: exit only at/below risk_exit and
            # after min_hold_events events since entry.  When the event
            # chain is exhausted the current state is terminal — there is
            # nothing left to oscillate against, so a settled safe state
            # satisfies the hold by definition (otherwise every case whose
            # final event is the recovery would deadlock in FORBIDDEN).
            held = self.event_count - self.entered_at_event_count
            if risk is not None and risk > self.risk_exit:
                self.zone_log.append({**entry, "decision": "held",
                                      "reason": "hysteresis",
                                      "held_events": held})
                return None
            if held < self.min_hold_events and not chain_exhausted:
                self.zone_log.append({**entry, "decision": "held",
                                      "reason": "min_hold",
                                      "held_events": held})
                return None
        self.zone = target
        if target == "FORBIDDEN":
            self.entered_at_event_count = self.event_count
        self.zone_log.append({**entry, "decision": "transitioned"})
        return f"{old}->{target}"

    # ---------------------------------------------------------- fast loop
    def on_environment_event(
        self,
        new_events: List[str],
        previous_facts: Dict[str, Any],
        current_facts: Dict[str, Any],
        intent: Any = None,
        chain_exhausted: bool = False,
    ) -> Optional[str]:
        """Assess a state change and return a boundary directive (or None).

        Silent unless the risk zone transitions or a quote expires — a state
        change that crosses no boundary produces no directive.
        """
        parts: List[str] = []
        transition = self._update_zone(current_facts,
                                       chain_exhausted=chain_exhausted)
        curr_risk = (current_facts or {}).get("risk_score")
        if transition is not None:
            new_zone = self.zone
            if new_zone == "FORBIDDEN":
                parts.append(
                    f"risk_score={curr_risk} >= enter threshold {self.risk_enter}: "
                    "execution is FORBIDDEN while the risk zone stays FORBIDDEN; "
                    "do not finalize EXECUTE — the correct terminal is BLOCK "
                    "(not REQUEST_CLARIFICATION): no user answer can lift an "
                    "environment-side prohibition."
                )
            elif transition.startswith("FORBIDDEN->"):
                parts.append(
                    f"risk_score={curr_risk} <= exit threshold {self.risk_exit}: "
                    "the execution prohibition is LIFTED, but the proposal must "
                    "be re-verified against the fresh state before any execution; "
                    "receipts bound to earlier state versions are stale."
                )
            elif new_zone == "OBSERVE":
                parts.append(
                    f"risk_score={curr_risk} entered the OBSERVE zone "
                    f"[{self.risk_exit}, {self.risk_enter}): proceed with "
                    "caution and re-verify the proposal against the fresh state "
                    "before finalizing."
                )
        quote_expired = bool(
            (previous_facts or {}).get("quote_valid")
            and not (current_facts or {}).get("quote_valid", True)
        )
        if quote_expired:
            if self.quote_policy == "forbid":
                parts.append(
                    "quote expired: execution is FORBIDDEN on the stale quote; "
                    "do not finalize EXECUTE until a fresh quote is available — "
                    "the correct terminal while the quote stays invalid is BLOCK."
                )
            else:
                parts.append(
                    "quote expired: re-query the current state and re-simulate the "
                    "proposal before any execution; receipts bound to earlier state "
                    "versions are stale."
                )
        # slow_loop: pressure bookkeeping rides the same event stream.
        if "slow_loop" in self.features:
            delta = -1  # calm event (or no transition): decay
            if transition is not None and self.zone == "FORBIDDEN":
                delta = +2
            elif transition is not None and self.zone == "OBSERVE":
                delta = +1
            if quote_expired:
                delta += 1
            escalated = self._pressure_event(delta)
            if escalated:
                parts.append(escalated)
        if not parts:
            return None
        directive = "H3 dynamic boundary directive: " + " ".join(parts)
        self.directives.append(directive)
        return directive

    # ---------------------------------------------------- feedback_replan
    #: failure_reason category -> replan guidance (decision table, keyed on
    #: the simulator's failure_reasons strings, not on case identities).
    _REPLAN_TABLE = (
        ("exceeds balance",
         "balance insufficient: retry with a reduced amount, or BLOCK"),
        ("exceeds max_amount",
         "amount above the per-transaction cap: reduce the amount below the "
         "cap and retry, or BLOCK"),
        ("quote is not valid",
         "quote invalid: obtain a fresh quote (re-query the current state) "
         "before simulating again"),
        ("no valid approval",
         "no valid approval/authorization: BLOCK"),
        ("exceeds max_slippage",
         "slippage above the allowed maximum: retry with a lower slippage "
         "parameter"),
        ("exceeds pool_liquidity",
         "pool liquidity insufficient: retry with a reduced amount, or BLOCK"),
    )

    def on_simulation_failure(self, receipt: Dict[str, Any]) -> Optional[str]:
        """feedback_replan hook: the runner calls this once per NEW rejected
        receipt (idempotent per receipt_id). Emits a category-keyed replan
        directive and records the proposal so decide() can reject a verbatim
        retry (same action + same canonical parameters)."""
        if "feedback_replan" not in self.features:
            return None
        if not isinstance(receipt, dict):
            return None
        receipt_id = receipt.get("receipt_id")
        if receipt_id and receipt_id in self._seen_failure_receipts:
            return None
        if receipt_id:
            self._seen_failure_receipts.append(receipt_id)
        reasons = receipt.get("failure_reasons") or []
        guidance = None
        for needle, text in self._REPLAN_TABLE:
            if any(needle in r for r in reasons):
                guidance = text
                break
        if guidance is None:
            guidance = "unclassified simulation failure: BLOCK or clarify"
        self._rejected_proposals.append({
            "action_id": receipt.get("action_id"),
            "parameters": receipt.get("parameters") or {},
        })
        # slow_loop: a rejected simulation is boundary pressure too.
        escalated = self._pressure_event(+1)
        parts = [
            f"simulation rejected ({'; '.join(reasons) or 'unknown reason'}): "
            f"{guidance}. Do NOT retry the identical proposal unchanged."
        ]
        if escalated:
            parts.append(escalated)
        directive = "H3 replan directive: " + " ".join(parts)
        self.directives.append(directive)
        return directive

    # ----------------------------------------------------- clarify_policy
    #: Fields whose answer can flip the feasibility simulation directly ask
    #: first; timing/preference fields ask later.
    _VOI_PRIORITY = ("operation", "asset", "amount", "sku", "quantity",
                     "target_asset", "payment_method", "max_total_usd",
                     "slippage", "deadline_at", "time_window_seconds")

    def clarify_directive(self, intent: Any) -> Optional[str]:
        """clarify_policy: value-of-information question ordering + budget.

        Returns a directive when the current intent carries multiple
        unresolved required fields; None when the feature is off, the intent
        is None, or fewer than two fields are unresolved. The budget counts
        clarification rounds: exhausted -> BLOCK, not another RC."""
        if "clarify_policy" not in self.features or intent is None:
            return None
        missing = list(getattr(intent, "missing_required", None) or [])
        pending = list(getattr(intent, "pending_resolution", None) or [])
        unresolved = missing + [f for f in pending if f not in missing]
        if len(unresolved) < 2:
            return None
        ordered = sorted(
            unresolved,
            key=lambda f: (self._VOI_PRIORITY.index(f)
                           if f in self._VOI_PRIORITY
                           else len(self._VOI_PRIORITY)),
        )
        feasibility_first = [f for f in ordered
                             if f in self._VOI_PRIORITY[:7]]
        directive = (
            "H3 clarification policy: multiple required fields are unresolved "
            f"({', '.join(unresolved)}). Ask in value-of-information order — "
            f"feasibility-deciding fields first ({', '.join(ordered)}); "
            f"the answer to each of {', '.join(feasibility_first) or 'them'} "
            "can flip the feasibility conclusion. Clarification budget: "
            f"{self._clarify_remaining} question(s) left; if the proposal is "
            "still infeasible when the budget is exhausted the correct "
            "terminal is BLOCK, not another REQUEST_CLARIFICATION."
        )
        self.directives.append(directive)
        return directive

    def note_clarification_round(self) -> None:
        """clarify_policy: the runner calls this when the team finalizes a
        REQUEST_CLARIFICATION — one budget unit is spent."""
        if "clarify_policy" in self.features and self._clarify_remaining > 0:
            self._clarify_remaining -= 1

    # ---------------------------------------------------------- slow loop
    def decide(
        self,
        final_decision: Dict[str, Any],
        environment_snapshot: Dict[str, Any],
        intent: Any = None,
        evidence: Any = None,
        ledger: Any = None,
    ) -> Dict[str, Any]:
        self.last_adjustment_reason = ""
        if not isinstance(final_decision, dict):
            return final_decision
        behavior = final_decision.get("final_behavior")
        # never upgrade a non-EXECUTE decision
        if behavior != "EXECUTE":
            return final_decision

        facts = (environment_snapshot or {}).get("facts", environment_snapshot or {})
        version = (environment_snapshot or {}).get("state_version")
        params = final_decision.get("parameters") or {}

        hard_block_reasons: List[str] = []
        if not facts.get("approval_present", True):
            hard_block_reasons.append("no valid approval/authorization")
        risk = facts.get("risk_score")
        if risk is not None and risk >= self.risk_enter:
            hard_block_reasons.append(
                f"risk_score {risk} >= enter threshold {self.risk_enter} "
                "(risk zone FORBIDDEN)"
            )
        elif self.zone == "FORBIDDEN" and risk is not None:
            # hysteresis: still inside the minimum-hold / above-exit window.
            # A settled terminal state (event chain exhausted) at/below the
            # exit threshold is not oscillation and does not block.
            _settled = (
                risk <= self.risk_exit
                and ledger is not None
                and getattr(ledger, "chain_exhausted", False)
            )
            if not _settled:
                hard_block_reasons.append(
                    f"risk zone FORBIDDEN (entered at risk >= {self.risk_enter}; "
                    f"current risk_score {risk} has not stayed <= exit threshold "
                    f"{self.risk_exit} for {self.min_hold_events} events)"
                )
        if facts.get("route_available") is False:
            hard_block_reasons.append("no route available")
        if self.quote_policy == "forbid" and facts.get("quote_valid") is False:
            # Learned quote stance "forbid": a stale quote is a hard
            # environment boundary, not merely a re-verification trigger.
            hard_block_reasons.append("quote is not valid (quote policy: forbid)")
        # slow_loop: an active escalation window downgrades EXECUTE to BLOCK.
        if "slow_loop" in self.features and self.escalation_active:
            hard_block_reasons.append(
                "boundary pressure escalation is active (slow_loop)")
        # feedback_replan: a verbatim retry of an already-rejected proposal
        # (same action + same parameters) is rejected outright.
        if "feedback_replan" in self.features:
            for rejected in self._rejected_proposals:
                if (rejected.get("action_id") == final_decision.get("action_id")
                        and rejected.get("parameters") == params):
                    hard_block_reasons.append(
                        "identical proposal was already rejected by the "
                        "simulator (feedback_replan: replan, do not retry "
                        "unchanged)")
                    break
        amount = params.get("amount")
        if isinstance(amount, (int, float)) and amount > facts.get(
            "balance", float("inf")
        ):
            hard_block_reasons.append("amount exceeds balance")
        if params.get("sku") is not None:
            # ecommerce-payment purchase decision: stock, seller risk and the
            # order-total caps are hard boundaries too.
            from payment.economics import purchase_breakdown

            product = next(
                (p for p in (environment_snapshot or {}).get("product_catalog", [])
                 if p.get("sku") == params.get("sku")),
                None,
            )
            if product is None:
                hard_block_reasons.append(f"unknown sku {params.get('sku')!r}")
            else:
                qty = params.get("quantity")
                stock = product.get("stock", 0)
                if (isinstance(qty, (int, float))
                        and isinstance(stock, (int, float)) and stock < qty):
                    hard_block_reasons.append(f"stock {stock} < quantity {qty}")
                seller_risk = product.get("seller_risk")
                threshold = facts.get("risk_threshold")
                if (isinstance(seller_risk, (int, float))
                        and isinstance(threshold, (int, float))
                        and seller_risk >= threshold):
                    hard_block_reasons.append(
                        f"seller risk {seller_risk} >= threshold {threshold}")
                total = purchase_breakdown(product, qty).get("total_price")
                if isinstance(total, (int, float)):
                    cap = facts.get("max_amount")
                    if isinstance(cap, (int, float)) and total > cap:
                        hard_block_reasons.append(
                            f"total_price {total} > per-order max {cap}")
                    bal = facts.get("balance")
                    if isinstance(bal, (int, float)) and total > bal:
                        hard_block_reasons.append(
                            f"total_price {total} > balance {bal}")

        if hard_block_reasons:
            return self._adjust(final_decision, "BLOCK",
                                "environment hard boundary: " + "; ".join(hard_block_reasons))

        # H1 fail-closed split (v4.2): essentially-infeasible intent (unsupported
        # asset/operation, expired deadline) -> BLOCK; merely unresolved fields
        # -> clarify. BLOCK outranks RC.
        if intent is not None and getattr(intent, "requires_block", False):
            missing = getattr(intent, "missing_required", None) or []
            return self._adjust(
                final_decision, "BLOCK",
                "intent is essentially infeasible (unsupported/expired): "
                + ", ".join(missing),
            )
        if intent is not None and (getattr(intent, "requires_clarification", False)
                                   or getattr(intent, "pending_resolution", None)):
            missing = (getattr(intent, "missing_required", None) or []) + (getattr(intent, "pending_resolution", None) or [])
            # clarify_policy: budget exhausted -> BLOCK, not another RC.
            if ("clarify_policy" in self.features
                    and self._clarify_remaining <= 0):
                return self._adjust(
                    final_decision, "BLOCK",
                    "clarification budget exhausted and the intent still has "
                    "unresolved required fields: " + ", ".join(missing),
                )
            return self._adjust(
                final_decision, "REQUEST_CLARIFICATION",
                "intent has unresolved required fields: " + ", ".join(missing),
            )

        # H2 hard gate veto: a veto whose failing checks are anything OTHER
        # than receipt binding is substantive (inviolable economic violation,
        # environment fact gate, catalog/schema mismatch) — no fresh
        # simulation can repair it, so EXECUTE is forbidden immediately,
        # BEFORE any re-verification top-up could whitewash the veto with a
        # new receipt.  Only a pure receipt-binding veto is repairable by
        # re-simulation below.
        if isinstance(evidence, dict) and evidence.get("hard_gate") == "REJECT":
            failing = [
                c for c in evidence.get("rule_checks", [])
                if c.get("verdict") == "FAIL"
            ]
            if any(c.get("check") != "receipt_binding" for c in failing):
                return self._adjust(
                    final_decision, "BLOCK",
                    "H2 hard-constraint gate vetoed the proposal: "
                    + "; ".join(
                        (c.get("reason") or c.get("check") or "")
                        for c in failing
                    ),
                )

        # EXECUTE requires a receipt bound to the CURRENT state version.
        # Re-verification comes BEFORE honoring an H2 veto: a receipt-binding
        # veto is exactly the failure this re-verification can repair.
        receipt_id = final_decision.get("receipt_id")
        action = next((a for a in (environment_snapshot or {}).get("action_catalog", [])
                       if a.get("id") == final_decision.get("action_id")), None)
        if (environment_snapshot or {}).get("action_catalog") and action is None:
            return self._adjust(final_decision, "BLOCK", "final action is not in current catalog")
        if action is not None and action.get("operation") != final_decision.get("operation"):
            return self._adjust(final_decision, "BLOCK", "final operation does not match action")
        context = ledger.execution_context() if hasattr(ledger, "execution_context") else None
        bound_ok = False
        if ledger is not None and receipt_id:
            for r in ledger.receipts:
                if r.get("receipt_id") == receipt_id:
                    bound_ok = receipt_bound_to_current(
                        r, state_version=version, intent=intent,
                        action_id=final_decision.get("action_id"), params=params,
                        param_spec=action.get("parameters", {}) if action else {},
                        execution_context=context,
                    )
                    break
        if not bound_ok and ledger is not None:
            action_id = final_decision.get("action_id")
            if action_id and ledger.sim_calls < ledger.sim_max_calls:
                before = len(ledger.receipts)
                ledger.simulate(action_id, params)
                if len(ledger.receipts) > before:
                    receipt = ledger.receipts[-1]
                    receipt["source"] = "h3_policy"
                    if receipt_bound_to_current(
                        receipt, state_version=ledger.current_version, intent=intent,
                        action_id=action_id, params=params,
                        param_spec=action.get("parameters", {}) if action else {},
                        execution_context=ledger.execution_context() if hasattr(ledger, "execution_context") else None,
                    ):
                        adjusted = dict(final_decision)
                        adjusted["receipt_id"] = receipt["receipt_id"]
                        self.last_adjustment_reason = (
                            "re-verified on current state version "
                            f"{receipt.get('state_version')} (receipt {receipt['receipt_id']})"
                        )
                        return adjusted

        # H2 hard gate veto -> forbid EXECUTE
        if (isinstance(evidence, dict) and evidence.get("hard_gate") == "REJECT"
                and not (bound_ok and any(c.get("verdict") == "FAIL" for c in evidence.get("rule_checks", [])) and all(
                    c.get("check") == "receipt_binding"
                    for c in evidence["rule_checks"] if c.get("verdict") == "FAIL"))):
            return self._adjust(final_decision, "BLOCK",
                                "H2 hard-constraint gate vetoed the proposal")

        if not bound_ok and ledger is not None:
            return self._adjust(
                final_decision, "BLOCK",
                "no receipt bound to the current state version and "
                "re-verification was impossible",
            )
        return final_decision

    def _adjust(self, decision: Dict[str, Any], behavior: str, reason: str) -> Dict[str, Any]:
        adjusted = dict(decision)
        adjusted["final_behavior"] = behavior
        if behavior != "EXECUTE":
            adjusted["receipt_id"] = None
        adjusted["rationale"] = (
            (decision.get("rationale") or "")
            + f"\n[H3 boundary adjustment -> {behavior}] {reason}"
        ).strip()
        self.last_adjustment_reason = f"{behavior}: {reason}"
        return adjusted


def directive_block(directives: List[str]) -> str:
    if not directives:
        return ""
    return "H3 dynamic boundary directives:\n" + "\n".join(
        f"- {d}" for d in directives
    )
