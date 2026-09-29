"""H1 IntentStructurer: LLM-based structured intent extraction + versioning.

Pipeline (048-style staging adapted to payment intents):
  1. scheme classification (schema-guided, 032);
  2. per-field reading with status + evidence + hardness (012/013 taxonomy);
  3. constraint split (hard vs soft) + conflict detection;
  4. deterministic post-checks in schema.finalize_intent (fail-closed,
     versioning, diff) — never delegated to the LLM.

The structurer is invoked once per case before the team is built, and again
whenever a user-role dynamic event is released mid-run (intent versioning:
changes to amount, recipient or authorization create a new version and invalidate the previous one).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from multiagent.llm import StructuredResponseError, classify_llm_error
from payment.h1.schema import (
    Commitment,
    FieldReading,
    StructuredIntent,
    finalize_intent,
)

from payment.verification_contract import VERIFICATION_CONTRACT

STRUCTURER_ATTEMPTS = 3


class IntentDraft(BaseModel):
    """LLM-facing structurer output schema: ONLY the fields the model should
    produce.  Versioning/diff/missing_required are computed deterministically
    by finalize_intent and never exposed to the model (exposing them caused
    retry-looping validation errors)."""

    scheme: str = "unknown"
    open_or_ambiguous: bool = False
    fields: Dict[str, FieldReading] = Field(default_factory=dict)
    commitments: List[Commitment] = Field(default_factory=list)
    hard_constraints: List[str] = Field(default_factory=list)
    soft_preferences: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    requires_clarification: bool = False
    clarification_question: Optional[str] = None


_SYSTEM = """Role: Intent Structurer
You convert a payment conversation into a structured intent object. Follow three stages:

Stage 1 — Scheme classification. Classify the request into exactly one scheme:
swap | deposit | withdraw | borrow | repay | transfer | purchase | unknown.
Different schemes have different required fields (given below); do not treat every fund request as "pay".

Stage 2 — Field reading. For EACH of the fields {operation, amount, asset, target_asset, slippage, time_window_seconds, deadline_at} output a reading. For a purchase scheme, ALSO read {sku (the exact catalog SKU the user's choice resolves to), quantity, payment_method, max_total_usd (a stated total budget cap in USD, if any)} with the same status taxonomy.
- value: the interpreted value (numbers as numbers; deadline_at as ISO-8601 WITH timezone, resolved against the conversation clock; null when not determinable);
- status: exactly one of
  explicit (the user stated it), inferred (derived from stated facts — say how in evidence),
  missing (required but absent), ambiguous (several readings possible),
  invalid (stated but unusable, e.g. amount=0), unsupported (outside the available operations/assets);
- evidence: the exact conversation span you relied on ("" when missing);
- hardness: "hard" for inviolable limits ("must not exceed", "at most", "no later than"), "soft" for preferences ("ideally", "at least around", "as soon as possible", approximations).
NEVER invent a value the user did not state or imply. Approximations ("about $953") keep status explicit with the stated number. "A bit of slippage is fine" without a number means slippage status=missing. An explicit upper bound ("below 2%", "at most 0.5%") IS the value (slippage=2.0 / 0.5 — slippage values are in PERCENT, hard). Amounts stated in USD ("$953 of USDT", "about $445k worth of ETH") mean amount = that USD number exactly as stated; never convert units or ask about conversion.

Purchase field vocabulary: payment_method is the canonical API identifier: "PayPal" -> "paypal", "credit card" -> "credit_card", "debit card" -> "debit_card". Preserve the original words in evidence. This spelling mapping does not grant authorization or select a different payment method; unknown or ambiguous methods remain unresolved. Never put a product display name into sku; an exact SKU must be grounded using observed product tools.

Stage 3 — Commitments (graded, never dual-listed). Represent ordinary operation/asset/product identity choices in fields, not as extra category=other commitments. For each additional constraint or preference the user expressed, output one entry in "commitments" with:
- rule: the requirement restated precisely (with the number if any);
- level — exactly one of:
  * inviolable: explicit must/must-not language ("必须", "最多不要超过", "at most", "never", "do not exceed") — the inviolable boundary;
  * strong_preference: a concrete value with softened framing;
  * aspirational_target: a directional target with hedged wording ("hopefully", "ideally", "at least around", "最好", "尽量") — optimize toward it, but it NEVER blocks or justifies clarification;
  * weak_preference: vague nice-to-haves;
- evidence: the exact span it came from;
- category — exactly one of: min_output (net proceeds / "receive at least X" in the incoming asset), total_fee (total fee cap), total_price (shopping order total cap in USD — shipping and coupons included), gas_cap (gas price cap in gwei), slippage_cap (slippage cap in percent — an executable catalog parameter, never "other"), amount_limit (amount bound, including "swap exactly X" with direction=exact, never "other"), deadline (time constraint), product_identity (an explicit exact product/variant restriction, verified by the tool-grounded sku field), other (an additional requirement outside the preceding supported representations; default when unsure);
- value/unit/direction when numeric (e.g. value=100, unit="RMB", direction="max"). The unit is mandatory whenever the value carries one (currency, seconds, percent, gwei) — never assume a unit conversion or compare different units without supported evidence;
Each requirement appears in EXACTLY ONE commitment — never duplicate an item across levels. Strength follows USER MODALITY, never catalog representability. Known categories stay in their named category regardless of catalog parameters: min_output, total_fee and total_price are receipt-verifiable, amount_limit is parameter-verifiable, and deadline uses top-level timing fields. Only a requirement outside these supported representations (e.g. "must use a private MEV route" or "execute at that rate") belongs to other; if explicit and hard, preserve it as inviolable with exact evidence. Merely optional execution hints remain weak_preference. A stated gas cap ("gas under 10 gwei") is verifiable against facts.gas_price_gwei; its level follows its modality ("keep gas under X" = inviolable bound).
List internal contradictions in conflicts. Set requires_clarification only when required information is genuinely missing/ambiguous/conflicting AND cannot be resolved from the environment snapshot below; write the exact clarification_question then.
- When the user corrects themselves in a later turn ("actually ...", "instead", "scratch that", "改为", "换成"), the LATEST statement supersedes the earlier one: update the field to the new value with evidence from the later turn, and do NOT record a conflict or set requires_clarification because of it. "conflicts" is only for contradictions the user has NOT resolved.
- Never set requires_clarification without writing a concrete clarification_question; an RC with no question to ask is a structuring error.

Environment snapshot (action/asset catalogs, network and conversation clock only — environment facts such as balances, prices and risk are deliberately NOT shown here; the team reads those via tools):
__ENVIRONMENT__

Use the snapshot actively while structuring:
- Resolve relative time expressions against conversation_start_at/timezone. Example: "by 12pm" with conversation_start_at 2026-08-30T08:38:08+08:00 means deadline_at = 2026-08-30T12:00:00+08:00 (status inferred); "within an hour" means time_window_seconds = 3600 (status explicit).
- Check the action catalog / asset catalog before marking anything unsupported.
- A catalog action's venue names the protocol it settles on; a user destination mention (e.g. "into Compound") is satisfied when it matches that venue, and is NOT an unresolved constraint in that case.
- Do NOT mark a field missing/ambiguous if the snapshot plus the conversation determine it.
- A purchase may identify a product by brand/model without spelling its SKU. Product tools are not yet available in this snapshot: read brand/model/category as additional fields, leave the SKU missing pending tool resolution, and do not ask the user for an internal SKU merely because the product catalog has not been queried. Use ambiguous only for ambiguity in the user's choice, not absent tool evidence.

Scheme-specific required fields:
__REQUIRED_TABLE__

Conversation clock: __CLOCK__
Available operations and assets come only from the environment snapshot above; requests outside them map to status=unsupported.
"""

_EVENT_ADDENDUM = """
The user has sent a NEW message after the earlier intent was structured:
__EVENTS__
Produce the UPDATED intent: carry over everything the new message does not
change, apply the changes it makes, and treat the new message as authoritative
where it contradicts the earlier intent (e.g. an asset substitution). Do not
keep stale values from the superseded intent.
IMPORTANT: a change the user just made is a RESOLVED revision, NOT a conflict —
do not list it in "conflicts" and do not set requires_clarification because of
it. "conflicts" is only for contradictions the user has NOT resolved.
The environment snapshot in this prompt reflects the CURRENT, post-event state
(it may legitimately contain new actions/assets introduced by the event).
"""


class H1IntentStructurer:
    """LLM-backed IntentStructurer with deterministic fail-closed post-checks."""

    def __init__(
        self,
        llm,
        *,
        catalog_required: Optional[Dict[str, List[str]]] = None,
        conversation_start_at: str = "",
        timezone: str = "",
        environment_snapshot: Optional[Dict[str, Any]] = None,
    ):
        self.llm = llm
        self.catalog_required = catalog_required or {}
        self.clock = f"{conversation_start_at} ({timezone})".strip()
        self.environment_snapshot = environment_snapshot or {}
        self._current: Optional[StructuredIntent] = None

    def configure(self, **case_context) -> None:
        """Per-case context injection (runner calls this before structure())."""
        self.catalog_required = case_context.get("catalog_required") or {}
        start = case_context.get("conversation_start_at") or ""
        tz = case_context.get("timezone") or ""
        self.clock = f"{start} ({tz})".strip()
        self.environment_snapshot = case_context.get("environment_snapshot") or {}
        self._current = None

    def update_context(self, **case_context) -> None:
        """Refresh the environment context WITHOUT resetting the version chain
        (used before re-structuring on a dynamic event: the post-event state
        may carry new catalog/asset entries that the revision relies on)."""
        if "catalog_required" in case_context:
            self.catalog_required = case_context["catalog_required"] or {}
        if "environment_snapshot" in case_context:
            self.environment_snapshot = case_context["environment_snapshot"] or {}

    # ------------------------------------------------------------- internals
    def _prompt(self, conversation: str, events: Optional[List[str]]) -> list:
        required_table = (
            json.dumps(self.catalog_required, ensure_ascii=False)
            if self.catalog_required
            else "(use the scheme defaults: swap needs amount/asset/target_asset/slippage; others need amount/asset)"
        )
        system = (
            _SYSTEM.replace("__REQUIRED_TABLE__", required_table)
            .replace("__CLOCK__", self.clock or "unknown")
            .replace(
                "__ENVIRONMENT__",
                json.dumps(self.environment_snapshot, ensure_ascii=False)
                if self.environment_snapshot
                else "(not available)",
            )
        )
        system += VERIFICATION_CONTRACT
        if events:
            system += _EVENT_ADDENDUM.replace("__EVENTS__", "\n".join(events))
        return [
            SystemMessage(content=system),
            HumanMessage(
                content="Conversation to structure:\n" + conversation
            ),
        ]
    def _call(self, conversation: str, events: Optional[List[str]]) -> StructuredIntent:
        chain = self.llm.with_structured_output(IntentDraft)
        last_exc: Optional[BaseException] = None
        for _ in range(STRUCTURER_ATTEMPTS):
            try:
                draft = chain.invoke(self._prompt(conversation, events))
                intent = StructuredIntent(**draft.model_dump())
                return finalize_intent(
                    intent,
                    catalog_required=self.catalog_required,
                    previous=self._current if events else None,
                    conversation=conversation,
                    conversation_start_at=(
                        self.environment_snapshot.get("conversation_start_at")
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - retried, then classified
                from multiagent.call_audit import record_structured_failure
                record_structured_failure(self.llm, "Intent Structurer", _ + 1, exc)
                last_exc = exc
                print(f"[h1-structurer] error, retrying: {exc}")
        raise StructuredResponseError(classify_llm_error(last_exc), str(last_exc))

    # ------------------------------------------------------------- protocol
    @property
    def current(self) -> Optional[StructuredIntent]:
        return self._current

    def structure(self, visible_conversation: str) -> StructuredIntent:
        from payment.h1.grounding import defer_sku
        self._conversation = visible_conversation
        self._current = defer_sku(self._call(visible_conversation, None))
        return self._current

    def on_user_event(
        self, user_events: List[str], visible_conversation: str
    ) -> Optional[StructuredIntent]:
        """Re-structure when new user turns were released; None if nothing new.

        The runner tracks which events were already processed and only calls
        this with genuinely new ones, so a call always produces a new version.
        """
        if not user_events:
            return None
        from payment.h1.grounding import defer_sku
        self._conversation = visible_conversation
        self._current = defer_sku(self._call(visible_conversation, user_events))
        return self._current

    def on_tool_evidence(self, observations, state_version):
        from payment.h1.grounding import ground_sku
        if self._current is None:
            return None
        updated = ground_sku(self._current, observations, self._conversation, state_version)
        if updated is not None:
            self._current = updated
        return updated

    def on_authorization_evidence(self, receipts, state_version):
        from payment.h1.grounding import resolve_authorization_commitments
        if self._current is None:
            return None
        updated = resolve_authorization_commitments(
            self._current, receipts, state_version)
        if updated is not None:
            self._current = updated
        return updated


def intent_injection_block(intent: StructuredIntent, events_pending: bool = False) -> str:
    """Text block injected into the B channel / initial prompt (B100).

    ``events_pending``: undelivered dynamic events may still revise the
    request, so requires_block is announced as PROVISIONAL and no terminal
    short-circuit order is planted — a stale "finalize BLOCK immediately"
    outlives the revision that lifted it (round7/round8 finding)."""
    block_rule = (
        "When requires_block=true the request is essentially infeasible (unsupported "
        "asset/operation or expired deadline) — the only acceptable outcome is BLOCK, "
        "never REQUEST_CLARIFICATION or EXECUTE."
        if not events_pending else
        "When requires_block=true the request is infeasible AS CURRENTLY SPECIFIED "
        "(unsupported asset/operation or expired deadline) — but undelivered "
        "environment events may still revise it, so treat the block as PROVISIONAL: "
        "engage the environment and let the event chain exhaust before finalizing."
    )
    lines = [
        "Current structured intent (v%d%s):"
        % (intent.version, ", supersedes v%d" % intent.supersedes if intent.supersedes else ""),
        json.dumps(intent.compact(), ensure_ascii=False),
        "Verification: inviolable is a strength, not a requirement for a catalog parameter. "
        "Apply the shared offline verification contract: min_output/total_fee/total_price "
        "use bound receipt values, amount_limit/slippage_cap use parameters, and deadline "
        "uses top-level timing fields. Unsupported hard requirements remain unresolved.",
        "Usage rules: fields with status missing/ambiguous/invalid/unsupported must NOT be "
        "guessed; commitment level 'inviolable' is inviolable; 'strong_preference' should be "
        "honored unless infeasible; 'aspirational_target' and 'weak_preference' guide "
        "optimization only and NEVER justify BLOCK or clarification by themselves. "
        + block_rule,
        "pending_resolution names fields that must be grounded with tools before finalization. "
        "Query the product tools to resolve brand/model to an exact SKU; do not ask the user "
        "for internal identifiers that a unique observed catalog match supplies. If tools "
        "cannot uniquely resolve the request, ask a targeted clarification. Never use a "
        "proposal or successful simulation as proof of the user's chosen identity.",
    ]
    if intent.requires_block and events_pending:
        lines.append(
            "PENDING EVENTS (intent v%d): requires_block=true at this version, "
            "but the event chain is not exhausted — a hidden revision may lift "
            "the infeasibility. Query the environment at least once and do NOT "
            "finalize BLOCK until no further events arrive."
            % intent.version
        )
    elif intent.requires_block:
        # Deterministic short-circuit (v4.2): an essentially-infeasible intent
        # has no correctable proposal, so the H2 veto contract's "revise at
        # least once" demand can only loop. State the endgame explicitly, but
        # bind it to THIS version: a user revision can lift the infeasibility
        # (e.g. swapping an unsupported asset for a supported one), and the
        # superseding version must void the order — a terminal-sounding stale
        # instruction otherwise outranks the revised intent (round7 finding).
        lines.append(
            "SHORT-CIRCUIT (intent v%d): requires_block=true — the request is "
            "infeasible as currently specified (%s). Do not generate or revise "
            "proposals and do not simulate; finalize BLOCK after at most one "
            "evaluation pass (an H2 veto on this path is satisfied by "
            "blocking). This instruction is bound to intent v%d ONLY: if a "
            "newer intent version supersedes it, this short-circuit is VOID — "
            "follow the latest version's status instead."
            % (intent.version, ", ".join(intent.missing_required or []),
               intent.version)
        )
    if intent.requires_clarification and not intent.requires_block:
        # Feasibility-first guard (round8 finding): these flags come from the
        # CONVERSATION alone (H1 is blind to account facts). When the real
        # infeasibility lives in the environment (balance/approval/limits),
        # an RC anchor suppresses the BLOCK path the team would otherwise
        # find — every clarification must first pass an environment check.
        lines.append(
            "Feasibility first (intent v%d): clarification is indicated, but "
            "before asking, DO THE MATH with environment facts: for each "
            "candidate interpretation, compute the best case (e.g. full "
            "balance x current spot rate vs a min_output commitment, cart "
            "total vs a hard cap, candidate amounts vs balance/limits). If NO "
            "candidate can satisfy the hard commitment, no answer will fix it "
            "— finalize BLOCK instead of REQUEST_CLARIFICATION. When the "
            "margin is close, confirm with a best-case simulation."
            % intent.version
        )
    if intent.diff_from_previous:
        lines.append("Changed fields vs previous version: " + json.dumps(intent.diff_from_previous, ensure_ascii=False))
        lines.append(
            "Any proposal or simulation receipt produced under the superseded version is VOID; "
            "regenerate the proposal from THIS version before evaluating or finalizing."
        )
    return "\n".join(lines)
