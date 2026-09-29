"""H2 AlignmentVerifier: hybrid rule+LLM verification of proposal alignment.

Layered verdict (Arbiter-style):
  * rule layer (deterministic, auditable): catalog membership, parameter
    schema, environment fact gates, receipt binding to the CURRENT state
    version (and intent version, when stamped), simulation-budget audit,
    forbidden-receipt check;
  * early termination: a critical rule failure skips the LLM layer;
  * semantic layer (LLM): whether the proposal faithfully realizes the
    conversation's intent (substitutions, skimming, loosened bounds); it runs
    only when the rule layer found a real proposal to verify (applicable);
    its REJECT verdict is honored only when double-signed (alignment_risk=
    high AND a verbatim evidence quote AND a structured mismatch passing the
    three-point deterministic check: proposal value present, latest-speaker
    rule over the user turns, values genuinely differ), otherwise downgraded
    to observations; semantic-layer LLM faults degrade to "unavailable" and
    never fail a case;
  * aggregation: any rule FAIL -> hard_gate=REJECT (veto contract); else a
    double-signed semantic REJECT vetoes; nothing to verify -> NOT_APPLICABLE.

The verifier never changes the team's decision itself: the evidence enters
the background (B) channel with an explicit veto contract the Revisor/Main
Supervisor are instructed to honor, and the runner enforces the veto
deterministically at FINISH (see runner's h2_enforcement record).
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from multiagent.llm import StructuredResponseError, classify_llm_error

from payment.verification_contract import ECONOMIC_FIELDS, VERIFICATION_CONTRACT

VERIFY_ATTEMPTS = 3
_REL_TOL = 1e-6
_ABS_TOL = 1e-9


class Mismatch(BaseModel):
    """One structured field-level contradiction claimed by the semantic judge."""

    field: str = ""
    reference_value: str = ""
    proposal_value: str = ""


class SemanticVerdict(BaseModel):
    """Semantic-layer verdict with a double-signed veto safety valve.

    A semantic REJECT is honored only when alignment_risk is "high", at least
    one evidence_quote passes a deterministic containment check against the
    conversation / structured-intent text, AND at least one structured
    mismatch passes the three-point deterministic verification (proposal
    value present in the proposal, latest-speaker rule over the user turns,
    and the two values genuinely differ).  A weak semantic judge must not be
    able to veto by itself — 037's finding is that LLM-only validation
    hallucinates technical facts."""

    verdict: str = Field(default="PASS", description="PASS | OBSERVATIONS | REJECT")
    observations: List[str] = Field(default_factory=list)
    alignment_risk: str = Field(default="low", description="low | medium | high")
    reason: str = ""
    evidence_quotes: List[str] = Field(
        default_factory=list,
        description="verbatim spans from the conversation or structured intent "
        "supporting the mismatch; REQUIRED for a REJECT verdict",
    )
    mismatches: List[Mismatch] = Field(
        default_factory=list,
        description="structured field-level contradictions (reference value vs "
        "proposal value); at least one REQUIRED for a REJECT verdict",
    )


def _normalize_quote(text: str) -> str:
    """Whitespace-collapsed, case-folded form for containment checks."""
    return " ".join(text.split()).lower()


def _quote_supported(quotes: List[str], *reference_texts: str) -> bool:
    """Deterministic containment check: at least one quote must be a verbatim
    (normalized) substring of a reference text."""
    haystacks = [_normalize_quote(t) for t in reference_texts if t]
    for quote in quotes:
        needle = _normalize_quote(quote)
        if needle and any(needle in h for h in haystacks):
            return True
    return False


def _values_differ(a: str, b: str) -> bool:
    """Number-aware inequality check on normalized surface forms
    (1290.0 == 1290.00)."""
    na, nb = _normalize_quote(a), _normalize_quote(b)
    if na == nb:
        return False
    try:
        return not math.isclose(float(na), float(nb),
                                rel_tol=_REL_TOL, abs_tol=_ABS_TOL)
    except (TypeError, ValueError):
        return True


def _user_turns(conversation: str) -> List[str]:
    """Extract user-role turns from the rendered conversation text."""
    turns: List[str] = []
    for line in conversation.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("user:"):
            turns.append(stripped)
    return turns


# Supersession / negation context patterns (case-insensitive; the value is
# re.escape'd; a few filler words such as articles may sit between the pattern
# word and the value).  A value occurrence inside any of these contexts no
# longer counts as the user's current intent.
_SUPERSESSION_PATTERNS = (
    r"instead\s+of\s+{gap}{value}",
    r"instead\s+for\s+{gap}{value}",
    r"rather\s+than\s+{gap}{value}",
    r"(?:do\s+not|don't|does\s+not|doesn't|never|not)\s+(?:\w+\s+){0,2}{value}",
    r"replace\s+{gap}{value}\s+with",
    r"switch\s+from\s+{gap}{value}",
    r"no\s+longer\s+{gap}{value}",
    r"stop\s+using\s+{gap}{value}",
)
_GAP = r"(?:the\s+|a\s+|that\s+|old\s+)*"


def _all_occurrences_superseded(turn: str, value: str) -> bool:
    """True when every occurrence of ``value`` in ``turn`` sits inside a
    supersession/negation context ("instead of USDC", "not DAI", ...)."""
    normalized = _normalize_quote(turn)
    needle = _normalize_quote(value)
    if not needle or needle not in normalized:
        return False
    superseded_spans: List[tuple] = []
    for pattern in _SUPERSESSION_PATTERNS:
        regex = pattern.replace("{gap}", _GAP).replace(
            "{value}", re.escape(needle))
        superseded_spans.extend(m.span() for m in re.finditer(regex, normalized))
    for occurrence in re.finditer(re.escape(needle), normalized):
        start = occurrence.start()
        if not any(s <= start < e for s, e in superseded_spans):
            return False  # an un-superseded reference survives
    return True


def _mismatch_verified(
    mismatch: "Mismatch", proposal: Dict[str, Any], conversation: str
) -> bool:
    """Three-point deterministic verification of a claimed mismatch:

    a. the proposal_value actually appears in the proposal JSON text;
    b. latest-speaker rule: among the user turns mentioning either value,
       the LAST one must mention the reference_value — and that mention must
       not be fully superseded/negated within the turn (a later revision
       mentioning only the proposal value, or mentioning the reference value
       only inside "instead of / not / replace ... with" contexts, voids the
       claimed mismatch);
    c. reference_value and proposal_value genuinely differ (number-aware).
    """
    reference = _normalize_quote(mismatch.reference_value)
    proposed = _normalize_quote(mismatch.proposal_value)
    if not reference or not proposed:
        return False
    # (c) the two values must genuinely differ
    if not _values_differ(mismatch.reference_value, mismatch.proposal_value):
        return False
    # (a) the proposal value must be present in the proposal JSON text
    proposal_text = _normalize_quote(json.dumps(proposal, ensure_ascii=False))
    if proposed not in proposal_text:
        return False
    # (b) latest-speaker rule over the user turns
    related = [
        t for t in _user_turns(conversation)
        if reference in _normalize_quote(t) or proposed in _normalize_quote(t)
    ]
    if not related:
        return False
    latest = related[-1]
    if reference not in _normalize_quote(latest):
        return False
    # (b+) supersession check: a reference value that appears ONLY inside
    # supersession/negation contexts in the latest turn is no longer the
    # user's intent and cannot support a veto.
    if _all_occurrences_superseded(latest, mismatch.reference_value):
        return False
    return True


def _num_close(a: Any, b: Any) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=_REL_TOL, abs_tol=_ABS_TOL)
    except (TypeError, ValueError):
        return False


def receipt_bound_to_current(
    receipt: Dict[str, Any],
    *,
    state_version: Any,
    intent: Any = None,
    action_id: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    param_spec: Optional[Dict[str, Any]] = None,
    execution_context: Optional[Dict[str, Any]] = None,
) -> bool:
    """Deterministic receipt-binding check shared by the H2 rule layer, the
    runner's H2 hard-gate enforcement and the H3 slow loop.

    A receipt is bound when it executed successfully on the CURRENT state
    version (and, when an H1 intent is present and the receipt carries an
    intent_version stamp, on the CURRENT intent version — receipts produced
    under a superseded intent are void).  When ``action_id``/``params`` are
    given, the receipt must also match the proposal's action and every
    catalog-spec parameter the proposal sets.  Receipts without an
    intent_version stamp (legacy/mock data) skip the intent check.
    """
    if receipt.get("status") != "SIMULATED_EXECUTED":
        return False
    if receipt.get("state_version") != state_version:
        return False
    if intent is not None and receipt.get("intent_version") is not None:
        intent_version = getattr(intent, "version", None)
        if intent_version is not None and receipt.get("intent_version") != intent_version:
            return False
    if action_id is not None and receipt.get("action_id") != action_id:
        return False
    if execution_context is not None:
        if not execution_context.get("approval_present"):
            return False
        for name in ("conversation_sha256", "authorization_version"):
            if receipt.get(name) != execution_context.get(name) or name not in receipt:
                return False
    if params is not None and param_spec is not None:
        rp = receipt.get("parameters", {})
        for name in param_spec:
            if name not in params and name not in rp:
                continue
            if name not in params or name not in rp:
                return False
            expected, got = params[name], rp[name]
            if isinstance(expected, (int, float)) or isinstance(got, (int, float)):
                if not _num_close(got, expected):
                    return False
            elif got != expected:
                return False
    return True


def _rule(name: str, ok: bool, reason: str = "") -> Dict[str, Any]:
    return {"check": name, "verdict": "PASS" if ok else "FAIL", "reason": reason}


class H2AlignmentVerifier:
    def __init__(self, llm):
        self.llm = llm
        self.conversation = ""
        self.sim_max_calls: Optional[int] = None

    def configure(self, **case_context) -> None:
        self.conversation = case_context.get("visible_conversation", "")
        self.sim_max_calls = case_context.get("simulation_max_calls")

    def update_conversation(self, conversation: str) -> None:
        """Refresh ONLY the conversation context (used when dynamic events are
        released mid-run: the verifier must judge against the extended
        conversation, not the stale first-turn one).  Other configured fields
        (simulation budget) stay untouched."""
        self.conversation = conversation

    # ------------------------------------------------------------- rule layer
    @staticmethod
    def _receipt_binding_diagnosis(
        successful: List[Dict[str, Any]],
        *,
        action_id: Any,
        state_version: Any,
        intent: Any = None,
    ) -> str:
        """Precise recovery guidance for a receipt_binding FAIL: distinguish
        'never simulated' / 'stale environment state' / 'voided by an intent
        revision' / 'parameters differ' so the team knows whether RE-RUNNING
        the simulation (not BLOCK) restores EXECUTE."""
        related = [r for r in successful if r.get("action_id") == action_id]
        if not related:
            return (
                "no simulation has been run for this action; run simulation "
                "first"
            )
        intent_version = getattr(intent, "version", None) if intent is not None else None
        if intent_version is not None and any(
            r.get("intent_version") is not None
            and r.get("intent_version") != intent_version
            for r in related
        ) and not any(
            r.get("intent_version") == intent_version for r in related
        ):
            return (
                "the proposal may still be valid; it was simulated under a "
                "superseded intent version — re-run simulation under the "
                "current intent version to obtain a fresh receipt, then "
                "EXECUTE is allowed"
            )
        if any(r.get("state_version") == state_version for r in related):
            # current-state receipts exist but none matches the parameters
            return (
                "no successful receipt bound to the current state version "
                "with matching action and parameters"
            )
        stale = max((r.get("state_version") for r in related), default=None)
        return (
            f"environment state has changed (receipt v{stale} vs current "
            f"v{state_version}): re-query the environment and re-run "
            "simulation"
        )

    def _rule_layer(
        self,
        proposal: Dict[str, Any],
        receipts: List[Dict[str, Any]],
        env: Dict[str, Any],
        recommendation: Optional[str],
        intent: Any = None,
    ) -> Dict[str, Any]:
        checks: List[Dict[str, Any]] = []
        action_id = proposal.get("action_id")
        params = proposal.get("parameters") or {}
        catalog = env.get("action_catalog", [])
        facts = env.get("facts", {})
        version = env.get("state_version")

        if not action_id:
            # no transaction proposed (BLOCK / clarification path): nothing to
            # gate.  (Mid-run receipts are normal evidence, never a violation;
            # the final-behavior receipt check belongs to the TSR scorer.)
            return {"checks": checks, "hard_gate": None, "applicable": False}

        action = next((a for a in catalog if a.get("id") == action_id), None)
        checks.append(_rule(
            "catalog_membership", action is not None,
            "" if action else f"action_id {action_id!r} not in the current action catalog",
        ))
        if action is None:
            return {"checks": checks, "hard_gate": "REJECT", "applicable": True}

        if action.get("operation") != proposal.get("operation"):
            checks.append(_rule(
                "operation_match", False,
                f"operation {proposal.get('operation')!r} != catalog operation "
                f"{action.get('operation')!r}",
            ))
        else:
            checks.append(_rule("operation_match", True))

        # parameter schema (same semantics as the simulator)
        spec = action.get("parameters", {})
        param_errors: List[str] = []
        for name, rules in spec.items():
            value = params.get(name)
            if value is None:
                if rules.get("required"):
                    param_errors.append(f"missing required parameter: {name}")
                continue
            if rules.get("type") == "number" and not isinstance(value, (int, float)):
                param_errors.append(f"{name} must be a number")
                continue
            if rules.get("type") == "string" and not isinstance(value, str):
                param_errors.append(f"{name} must be a string")
                continue
            if rules.get("type") == "number":
                if "minimum" in rules and value < rules["minimum"]:
                    param_errors.append(f"{name}={value} below minimum {rules['minimum']}")
                if "maximum" in rules and value > rules["maximum"]:
                    param_errors.append(f"{name}={value} above maximum {rules['maximum']}")
        checks.append(_rule("parameter_schema", not param_errors,
                            "; ".join(param_errors)))

        # Intent<->proposal field alignment (certification-chain hop 1) —
        # only when H1's structured intent is available (B110+).  Deterministic:
        # a proposal field contradicting an explicit/inferred intent reading is
        # a hard FAIL; fields in the intent's missing_required list must not be
        # silently filled (optional fields the intent lacks may be supplied).
        if intent is not None and hasattr(intent, "field"):
            intent_errors: List[str] = []
            missing_required = set(getattr(intent, "missing_required", None) or [])
            missing_required.update(getattr(intent, "pending_resolution", None) or [])
            proposal_view = dict(params)
            for extra in ("time_window_seconds", "deadline_at"):
                if extra in proposal and extra not in proposal_view:
                    proposal_view[extra] = proposal[extra]
            for name in ("operation", "amount", "asset", "target_asset",
                         "slippage", "time_window_seconds", "deadline_at",
                         "sku", "quantity", "payment_method"):
                reading = intent.field(name)
                expected = reading.value
                if name == "operation":
                    got = proposal.get("operation")
                else:
                    got = proposal_view.get(name)
                if name in missing_required and got is not None:
                    intent_errors.append(
                        f"{name}={got!r} was filled although the intent "
                        f"requires clarification for it (missing)"
                    )
                    continue
                if reading.status in ("missing", "ambiguous", "invalid",
                                      "unsupported"):
                    continue
                if expected is None or got is None:
                    continue
                if isinstance(expected, (int, float)) or isinstance(got, (int, float)):
                    if not _num_close(got, expected):
                        intent_errors.append(
                            f"{name}: proposal {got!r} != intent {expected!r}"
                        )
                elif str(got) != str(expected):
                    intent_errors.append(
                        f"{name}: proposal {got!r} != intent {expected!r}"
                    )
            checks.append(_rule("intent_field_alignment", not intent_errors,
                                "; ".join(intent_errors)))

        fact_errors: List[str] = []
        if not facts.get("approval_present", False):
            fact_errors.append("no valid approval/authorization")
        amount = params.get("amount")
        if isinstance(amount, (int, float)):
            if amount > facts.get("max_amount", float("inf")):
                fact_errors.append(f"amount {amount} > max_amount {facts.get('max_amount')}")
            if amount > facts.get("balance", float("inf")):
                fact_errors.append(f"amount {amount} > balance {facts.get('balance')}")
        slippage = params.get("slippage")
        if isinstance(slippage, (int, float)) and slippage > facts.get("max_slippage", 100.0):
            fact_errors.append(f"slippage {slippage}% > max_slippage {facts.get('max_slippage')}%")
        gas_cap = params.get("max_gas_gwei")
        gas_now = facts.get("gas_price_gwei")
        if (isinstance(gas_cap, (int, float)) and isinstance(gas_now, (int, float))
                and gas_cap < gas_now):
            fact_errors.append(
                f"gas price {gas_now} gwei exceeds cap {gas_cap} gwei"
            )
        if not facts.get("quote_valid", True):
            fact_errors.append("quote invalid")
        if not facts.get("plan_valid", True):
            fact_errors.append("plan invalid")
        if not facts.get("route_available", True):
            fact_errors.append("no route available")
        if action.get("operation") == "purchase":
            # ecommerce-payment gates mirror the simulator: catalog
            # membership, stock, seller policy and the order-total caps.
            from payment.economics import purchase_breakdown

            product = next(
                (p for p in env.get("product_catalog", [])
                 if p.get("sku") == params.get("sku")),
                None,
            )
            if product is None:
                fact_errors.append(f"unknown sku {params.get('sku')!r}")
            else:
                qty = params.get("quantity")
                stock = product.get("stock", 0)
                if (isinstance(qty, (int, float))
                        and isinstance(stock, (int, float)) and stock < qty):
                    fact_errors.append(f"stock {stock} < quantity {qty}")
                seller = product.get("seller")
                if seller in (facts.get("blocked_sellers") or []):
                    fact_errors.append(f"seller {seller!r} is blocked")
                seller_risk = product.get("seller_risk")
                threshold = facts.get("risk_threshold")
                if (isinstance(seller_risk, (int, float))
                        and isinstance(threshold, (int, float))
                        and seller_risk >= threshold):
                    fact_errors.append(
                        f"seller risk {seller_risk} >= threshold {threshold}")
                allowed_categories = facts.get("allowed_categories") or []
                if (allowed_categories
                        and product.get("category") not in allowed_categories):
                    fact_errors.append(
                        f"category {product.get('category')!r} not allowed")
                total = purchase_breakdown(product, qty).get("total_price")
                if isinstance(total, (int, float)):
                    cap = facts.get("max_amount")
                    if isinstance(cap, (int, float)) and total > cap:
                        fact_errors.append(
                            f"total_price {total} > per-order max {cap}")
                    bal = facts.get("balance")
                    if isinstance(bal, (int, float)) and total > bal:
                        fact_errors.append(
                            f"total_price {total} > balance {bal}")
        checks.append(_rule("environment_facts", not fact_errors, "; ".join(fact_errors)))

        # M6 economic consistency: when the intent carries a verifiable
        # economic commitment (min_output / total_fee) and a successful
        # receipt exists, the receipt's economics must honor it. Without a
        # matching commitment the check passes (not applicable).
        matching_receipts = [r for r in receipts if receipt_bound_to_current(
            r, state_version=version, intent=intent, action_id=action_id,
            params=params, param_spec=spec)]
        checks.append(self._economic_consistency(matching_receipts, intent, facts))
        unresolved = getattr(intent, "unresolved_hard_constraints", None) or []
        if unresolved:
            checks.append(_rule("unresolved_hard_constraint", False, "; ".join(unresolved)))

        # receipt binding to the CURRENT state version — only meaningful when
        # the proposal is headed for EXECUTION (a BLOCK/CLARIFICATION path has
        # no obligation to hold a receipt).
        if recommendation == "EXECUTE":
            successful = [r for r in receipts if r.get("status") == "SIMULATED_EXECUTED"]

            def receipt_matches(r: Dict[str, Any]) -> bool:
                return receipt_bound_to_current(
                    r,
                    state_version=version,
                    intent=intent,
                    action_id=action_id,
                    params=params,
                    param_spec=spec,
                )

            bound = [r for r in successful if receipt_matches(r)]
            binding_reason = ""
            if not bound:
                binding_reason = self._receipt_binding_diagnosis(
                    successful, action_id=action_id, state_version=version,
                    intent=intent,
                )
            checks.append(_rule(
                "receipt_binding", bool(bound),
                "" if bound else binding_reason,
            ))

        if self.sim_max_calls is not None:
            n_calls = len([r for r in receipts])  # receipts exist only for executed sims
            checks.append(_rule(
                "simulation_budget", n_calls <= self.sim_max_calls,
                "" if n_calls <= self.sim_max_calls else
                f"{n_calls} simulations exceed budget {self.sim_max_calls}",
            ))

        hard_gate = "REJECT" if any(c["verdict"] == "FAIL" for c in checks) else None
        return {"checks": checks, "hard_gate": hard_gate, "applicable": True}

    @staticmethod
    def _subject(proposal, receipts, env, intent):
        """The exact inputs covered by an evidence record, never a model claim."""
        import hashlib
        payload = {
            "proposal": proposal, "receipts": receipts, "environment": env,
            "intent": intent.compact() if hasattr(intent, "compact") else None,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str,
                                         ensure_ascii=False).encode()).hexdigest()

    def revalidate_final(self, decision, receipts, environment_snapshot,
                         intent=None, evidence=None, execution_context=None):
        """Recheck FINISH after revisions; reuse only evidence for identical inputs."""
        proposal = {k: decision.get(k) for k in (
            "action_id", "operation", "parameters", "time_window_seconds", "deadline_at")}
        subject = self._subject(dict(proposal, receipt_id=decision.get("receipt_id")), receipts,
                                dict(environment_snapshot, execution_context=execution_context), intent)
        if (isinstance(evidence, dict) and evidence.get("final_subject_sha256") == subject
                and evidence.get("checked_recommendation") == "EXECUTE"):
            return evidence
        checked = self.verify({"proposal": proposal, "decision_recommendation": "EXECUTE"},
                              receipts, environment_snapshot, intent)
        action = next((a for a in environment_snapshot.get("action_catalog", [])
                       if a.get("id") == proposal["action_id"]), None)
        cited = next((r for r in receipts if r.get("receipt_id") == decision.get("receipt_id")), None)
        bound = action is not None and cited is not None and receipt_bound_to_current(
            cited, state_version=environment_snapshot.get("state_version"), intent=intent,
            action_id=proposal["action_id"], params=proposal["parameters"],
            param_spec=action.get("parameters", {}), execution_context=execution_context)
        if not bound:
            checked["rule_checks"].append(_rule("receipt_binding", False,
                "final decision does not explicitly cite a fully bound receipt"))
            checked["hard_gate"] = checked["overall"] = "REJECT"
        checked["final_subject_sha256"] = subject
        return checked

    # ----------------------------------------------------- economic consistency
    @staticmethod
    def _economic_consistency(receipts: List[Dict[str, Any]],
                              intent: Any, facts=None) -> Dict[str, Any]:
        """M6 rule: receipt economics vs the intent's economic commitments.

        Only commitments with ``category`` in {min_output, total_fee,
        total_price} are verifiable against receipt economics. Among those,
        ONLY ``level == "inviolable"`` enters the hard gate: for (min_output, min,
        V) the latest successful receipt's ``estimated_output`` must be >= V;
        for (total_fee, max, V) its ``total_fee`` must be <= V (both with
        _num_close tolerance). A violated ``strong_preference`` economic
        commitment is reported as an observation (recorded in the reason,
        never a FAIL); ``aspirational_target``/``weak_preference`` never
        block. No matching commitment or no successful receipt is not applicable.
        A mismatched unit requires same-state fee conversion evidence before
        comparison; unverified conversion never certifies a hard commitment.
        """
        commitments = list(getattr(intent, "commitments", None) or [])
        relevant = [c for c in commitments
                    if getattr(c, "category", None) in ECONOMIC_FIELDS]
        successful = [r for r in receipts
                      if r.get("status") == "SIMULATED_EXECUTED"]
        receipt = successful[-1] if successful else {}
        errors: List[str] = []
        observations: List[str] = []
        evidence: List[Dict[str, Any]] = []
        for c in relevant:
            field, direction = ECONOMIC_FIELDS[c.category]
            value = getattr(c, "value", None)
            unit = getattr(c, "unit", "") or ""
            level = getattr(c, "level", None)
            got = receipt.get(field)
            record = {"category": c.category, "level": level,
                      "receipt_id": receipt.get("receipt_id"),
                      "field": field, "observed": got, "bound": value,
                      "direction": direction, "unit": unit,
                      "status": "not_verified"}
            evidence.append(record)
            if not isinstance(value, (int, float)) or got is None:
                record["reason"] = "missing bound or receipt value; PASS is not proof"
                continue
            from payment.quantities import receipt_quantity, finite
            converted, conversion = receipt_quantity(receipt, field, unit,
                (facts or {}).get("spot_prices") if facts is not None else None)
            record["conversion"] = conversion
            record["receipt_unit"] = conversion.get("source_unit")
            comparable = (finite(value) and converted is not None
                          and getattr(c, "direction", None) in (None, direction))
            if not comparable:
                record["reason"] = conversion.get("reason", "invalid bound/direction")
                if level == "inviolable":
                    errors.append(f"unverified {field}: " + record["reason"])
                continue
            got = converted
            record["observed_in_bound_unit"] = got
            ok = ((got >= value if direction == "min" else got <= value) or _num_close(got, value))
            record["status"] = "satisfied" if ok else "violated"
            if not ok:
                label = {"min_output": "min_output commitment",
                         "total_fee": "fee cap commitment",
                         "total_price": "budget cap commitment"}[c.category]
                operator = "<" if direction == "min" else ">"
                msg = f"{field} {got} {operator} {label} {value} {unit}"
                if level == "inviolable":
                    errors.append(msg)
                elif level == "strong_preference":
                    observations.append(msg + " [strong_preference]")
        result = _rule("economic_consistency", not errors,
                       "; ".join(errors if errors else observations))
        result["commitment_evidence"] = evidence
        return result

    # ------------------------------------------------------------- semantic layer
    def _semantic_layer(self, proposal: Dict[str, Any], receipts: List[Dict[str, Any]],
                        rule_report: Dict[str, Any],
                        intent: Any = None) -> SemanticVerdict:
        if intent is not None and hasattr(intent, "compact"):
            reference_block = (
                "The authoritative field reference is the STRUCTURED INTENT below "
                "(field status explicit/inferred = honor it; missing/ambiguous/"
                "invalid/unsupported = the proposal must not invent it). "
                "Cross-check the proposal against the intent fields, not the raw "
                "wording."
            )
        else:
            reference_block = (
                "No structured intent is available; cross-check the proposal "
                "against the raw conversation below."
            )
        system = (
            "Role: Alignment Verifier\n"
            "You are an independent verifier. The deterministic rule checks have "
            "already passed — do not re-litigate numbers they covered.\n"
            + reference_block
            + "\nReport ONLY concrete, quotable candidate mismatches between the "
            "reference and the proposal fields (wrong operation/asset/"
            "amount/slippage/timing, a substituted value, or a value the "
            "reference does not support presented as fact), as 'observations'. "
            "Optional execution hints and preferences are not mismatches. "
            "An explicit must/must-not requirement remains binding even if the "
            "catalog cannot represent it; never turn an unsupported hard "
            "execution requirement into a soft preference. Approximate phrasing and soft "
            "preferences are normal and are NOT mismatches. If nothing concrete "
            "exists, return verdict=PASS, an empty observations list and "
            "alignment_risk=low. Use medium/high only with a concrete quoted "
            "mismatch. "
            "Return verdict=OBSERVATIONS for candidate mismatches the team "
            "should double-check. Return verdict=REJECT only when the mismatch "
            "is a definite, execution-critical contradiction (alignment_risk "
            "MUST be 'high') AND you can prove it: fill evidence_quotes with "
            "the exact spans copied from the conversation or the structured "
            "intent that the proposal contradicts, AND fill mismatches with at "
            "least one structured entry {field, reference_value, "
            "proposal_value} naming the field, the value the reference "
            "requires, and the value the proposal actually carries. The "
            "reference_value MUST come from the LAST user turn that discusses "
            "the field — when the user revised a value mid-conversation, only "
            "the latest revision counts; quoting a superseded earlier turn "
            "voids the mismatch. A REJECT without verbatim supporting quotes "
            "and a verifiable structured mismatch is automatically downgraded."
        )
        system += VERIFICATION_CONTRACT
        payload: Dict[str, Any] = {
            "proposal": proposal,
            "receipts": receipts,
            "rule_checks": rule_report.get("checks", []),
        }
        if intent is not None and hasattr(intent, "compact"):
            payload["structured_intent"] = intent.compact()
        payload["conversation"] = self.conversation
        user = json.dumps(payload, ensure_ascii=False)
        chain = self.llm.with_structured_output(SemanticVerdict)
        last_exc: Optional[BaseException] = None
        for _ in range(VERIFY_ATTEMPTS):
            try:
                return chain.invoke(
                    [SystemMessage(content=system), HumanMessage(content=user)]
                )
            except Exception as exc:  # noqa: BLE001 - retried, then classified
                from multiagent.call_audit import record_structured_failure
                record_structured_failure(self.llm, "Alignment Verifier", _ + 1, exc)
                last_exc = exc
                print(f"[h2-verifier] error, retrying: {exc}")
        raise StructuredResponseError(classify_llm_error(last_exc), str(last_exc))

    # ------------------------------------------------------------- protocol
    def verify(
        self,
        intermediate_output: Any,
        receipts: List[Dict[str, Any]],
        environment_snapshot: Dict[str, Any],
        intent: Any = None,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(intermediate_output, dict):
            return None
        proposal = intermediate_output.get("proposal")
        if not isinstance(proposal, dict):
            return None
        final_decision = intermediate_output.get("final_decision")
        recommendation = (
            (final_decision.get("final_behavior") if isinstance(final_decision, dict) else None)
            or intermediate_output.get("decision_recommendation")
        )

        rule_report = self._rule_layer(proposal, receipts, environment_snapshot,
                                       recommendation, intent)
        semantic: Optional[SemanticVerdict] = None
        semantic_status = "skipped"
        semantic_error = ""
        if rule_report["applicable"] and rule_report["hard_gate"] is None:
            # rules pass on a real proposal -> semantic layer runs (nothing to
            # verify on a BLOCK/RC path: applicable=False skips it entirely)
            try:
                semantic = self._semantic_layer(proposal, receipts, rule_report,
                                                intent)
                semantic_status = "ok"
            except Exception as exc:  # noqa: BLE001 - degraded, never raised
                # LLM faults must never fail the whole case: the rule layer
                # alone decides the outcome.
                semantic_status = "unavailable"
                semantic_error = f"{type(exc).__name__}: {exc}"

        # Double-signed semantic veto: a semantic REJECT stands only when the
        # risk is high, at least one evidence quote verifies verbatim against
        # the conversation / structured intent, AND at least one structured
        # mismatch passes the three-point deterministic verification
        # (proposal value present, latest-speaker rule, values differ);
        # otherwise it is downgraded to observations.
        semantic_veto = False
        semantic_veto_downgraded = False
        semantic_veto_downgrade_reason = ""
        if (
            semantic is not None
            and semantic.verdict == "REJECT"
            and rule_report["hard_gate"] is None
        ):
            intent_text = (
                json.dumps(intent.compact(), ensure_ascii=False)
                if intent is not None and hasattr(intent, "compact")
                else ""
            )
            if semantic.alignment_risk != "high":
                semantic_veto_downgraded = True
                semantic_veto_downgrade_reason = (
                    f"semantic REJECT with alignment_risk={semantic.alignment_risk!r} "
                    "(only 'high' may veto)"
                )
            elif not _quote_supported(semantic.evidence_quotes,
                                      self.conversation, intent_text):
                semantic_veto_downgraded = True
                semantic_veto_downgrade_reason = (
                    "no evidence_quote verified verbatim against the "
                    "conversation or the structured intent"
                )
            elif not semantic.mismatches:
                semantic_veto_downgraded = True
                semantic_veto_downgrade_reason = (
                    "semantic REJECT without any structured mismatch entry"
                )
            elif not any(
                _mismatch_verified(m, proposal, self.conversation)
                for m in semantic.mismatches
            ):
                semantic_veto_downgraded = True
                semantic_veto_downgrade_reason = (
                    "no mismatch passed the three-point verification "
                    "(proposal-value presence, latest-speaker rule, "
                    "values differ)"
                )
            else:
                semantic_veto = True

        if rule_report["hard_gate"] == "REJECT":
            overall = "REJECT"
        elif not rule_report["applicable"]:
            overall = "NOT_APPLICABLE"
        elif semantic_veto:
            overall = "REJECT"
        elif semantic is not None and (
            semantic.observations or semantic_veto_downgraded
            or semantic.verdict == "OBSERVATIONS"
        ):
            overall = "PASS_WITH_OBSERVATIONS"
        else:
            overall = "PASS"

        return {
            "overall": overall,
            "hard_gate": rule_report["hard_gate"],
            "rule_checks": rule_report["checks"],
            "semantic": (
                {
                    "verdict": semantic.verdict,
                    "observations": semantic.observations,
                    "alignment_risk": semantic.alignment_risk,
                    "reason": semantic.reason,
                    "evidence_quotes": semantic.evidence_quotes,
                    "mismatches": [m.model_dump() for m in semantic.mismatches],
                }
                if semantic is not None
                else None
            ),
            "semantic_status": semantic_status,
            "semantic_error": semantic_error,
            "semantic_veto": semantic_veto,
            "semantic_veto_downgraded": semantic_veto_downgraded,
            "semantic_veto_downgrade_reason": semantic_veto_downgrade_reason,
            "state_version": environment_snapshot.get("state_version"),
            "subject_sha256": self._subject(proposal, receipts, environment_snapshot, intent),
            "checked_recommendation": recommendation,
        }


def evidence_block(evidence: Optional[Dict[str, Any]]) -> str:
    """Text block injected into the B channel when H2 evidence exists."""
    if not evidence:
        return ""
    lines = [
        "Independent alignment verification (H2):",
        json.dumps(evidence, ensure_ascii=False),
    ]
    if evidence.get("hard_gate") == "REJECT":
        lines.append(
            "VETO CONTRACT: the deterministic hard-constraint gate REJECTED this "
            "proposal; EXECUTE is forbidden until the proposal is corrected and "
            "re-verified. When the failure is a correctable field-level mismatch "
            "(wrong parameter, substituted value), FIRST revise the proposal and "
            "resubmit it for verification (make at least one revision attempt). "
            "When the failure is receipt staleness (state version changed or the "
            "receipt was produced under a superseded intent version), the "
            "recovery is to RE-RUN the simulation on the current state — do NOT "
            "change the proposal parameters for that. "
            "Choose BLOCK only when execution is fundamentally infeasible or "
            "disallowed (insufficient balance, missing approval, risk boundary, "
            "unsatisfiable user requirement), and REQUEST_CLARIFICATION when "
            "required information is missing. "
            "When the intent has requires_block=true the request is fundamentally "
            "infeasible: there is no correctable proposal, so the revision attempt "
            "is WAIVED — finalize BLOCK immediately instead of revising."
        )
    elif evidence.get("semantic_veto"):
        lines.append(
            "VETO CONTRACT: the independent semantic verifier REJECTED this "
            "proposal with a high-risk, quote-supported mismatch (see "
            "evidence_quotes); EXECUTE is forbidden until the proposal is "
            "corrected and re-verified. FIRST revise the proposal to match the "
            "quoted reference and resubmit it for verification (make at least "
            "one revision attempt). Choose BLOCK only when execution is "
            "fundamentally infeasible or disallowed, and "
            "REQUEST_CLARIFICATION when required information is missing. "
            "When the intent has requires_block=true the request is fundamentally "
            "infeasible: the revision attempt is WAIVED — finalize BLOCK "
            "immediately instead of revising."
        )
    return "\n".join(lines)
