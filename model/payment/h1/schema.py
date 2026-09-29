"""H1 structured-intent schema.

A StructuredIntent is the field-level, verifiable representation of the
user's payment goal.  Every field carries:
  * value      — the interpreted value (None when not determinable);
  * status     — explicit | inferred | missing | ambiguous | invalid |
                 unsupported (AwN issue taxonomy mapped onto fields);
  * evidence   — the conversation span the reading came from ("" if none);
  * hardness   — hard (inviolable constraint) | soft (preference).

Time rule (fixed contract): time expressions must resolve against the
runner-provided conversation clock (conversation_start_at + timezone) into
ISO-8601-with-timezone values; underivable expressions keep status
missing/ambiguous and never leak raw text into value.

RC/BLOCK split (v4.2, deterministic — never LLM-reported): unresolved
required fields are graded by whether clarification can help. ``unsupported``
(catalog-foreign asset/operation) and an expired ``deadline_at`` (invalid,
predating the conversation clock) are essentially infeasible — no user answer
repairs them — so they set ``requires_block`` (BLOCK outranks RC when both
fire). ``missing``/``ambiguous`` and the other ``invalid`` shapes (amount=0
is treated as missing by the dataset validator) keep ``requires_clarification``.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from payment.h1.payment_method import canonical_payment_method

FIELD_STATUS = ("explicit", "inferred", "missing", "ambiguous", "invalid", "unsupported")

# Commitment levels (graded; modality-driven, never dual-listed):
#   inviolable          — Absolute, mandatory, upper-bound or prohibition wording; enforced by the H2 hard gate.
#   strong_preference   — Explicit numeric preference expressed with softened wording.
#   aspirational_target — Directional aspirations such as "prefer" or "ideally"; never block execution.
#   weak_preference     — Vague preference.
COMMITMENT_LEVELS = (
    "inviolable",
    "strong_preference",
    "aspirational_target",
    "weak_preference",
)

# Commitment categories (what the constraint is ABOUT; model-filled at draft
# time).  The H2 economic_consistency gate only fires on the economic ones
# (min_output / total_fee / total_price); the rest are informational for
# routing/audit.
COMMITMENT_CATEGORIES = (
    "min_output",     # Minimum net amount of the incoming asset.
    "total_fee",      # Maximum total fee.
    "total_price",    # Maximum order total in USD, including shipping and discounts.
    "gas_cap",        # Maximum gas price in gwei.
    "slippage_cap",   # Maximum slippage in percent; a required catalog parameter.
    "amount_limit",   # Upper or lower amount bound.
    "deadline",       # Time constraint.
    "product_identity",  # exact product/variant, grounded by observed shop tools
    "other",          # Other constraints (default).
)


class Commitment(BaseModel):
    """A single constraint/preference with graded strength, unit and direction."""

    rule: str
    level: Literal[
        "inviolable", "strong_preference", "aspirational_target", "weak_preference"
    ] = "weak_preference"
    category: Literal[
        "min_output", "total_fee", "total_price", "gas_cap", "slippage_cap",
        "amount_limit", "deadline", "product_identity", "other"
    ] = "other"
    evidence: str = ""
    value: Optional[float] = None
    unit: Optional[str] = None
    direction: Optional[str] = None  # "max" | "min" | "exact" | None
    note: str = ""
    # Python-computed (never model-filled): whether the evidence span was
    # verified against the conversation text by finalize_intent.
    evidence_verified: bool = True

# Canonical field set, aligned with the dataset's requested_action_contract.
INTENT_FIELDS = (
    "operation",
    "amount",
    "asset",
    "target_asset",
    "slippage",
    "time_window_seconds",
    "deadline_at",
)

# scheme -> required fields (schema-guided layer; the runner extends this with
# the live action-catalog parameter requirements).
SCHEME_REQUIRED_FIELDS: Dict[str, tuple] = {
    "swap": ("amount", "asset", "target_asset", "slippage"),
    "deposit": ("amount", "asset"),
    "withdraw": ("amount", "asset"),
    "borrow": ("amount", "asset"),
    "repay": ("amount", "asset"),
    "transfer": ("amount", "asset"),
    "purchase": ("sku", "quantity", "payment_method"),
    "unknown": ("amount",),
}


class FieldReading(BaseModel):
    value: Optional[Any] = None
    status: Literal[
        "explicit", "inferred", "missing", "ambiguous", "invalid", "unsupported"
    ] = "missing"
    evidence: str = ""
    hardness: Literal["hard", "soft"] = "soft"
    # Python-computed (never model-filled): whether the evidence span was
    # verified against the conversation text by finalize_intent.
    evidence_verified: bool = True
    # Python-computed (never model-filled): True when a later user turn
    # resolved this field (previous status missing/ambiguous/invalid/
    # unsupported -> now explicit/inferred).
    confirmed: bool = False


class StructuredIntent(BaseModel):
    version: int = 1
    scheme: str = "unknown"
    open_or_ambiguous: bool = False
    fields: Dict[str, FieldReading] = Field(default_factory=dict)
    commitments: List[Commitment] = Field(default_factory=list)
    hard_constraints: List[str] = Field(default_factory=list)
    soft_preferences: List[str] = Field(default_factory=list)
    missing_required: List[str] = Field(default_factory=list)
    pending_resolution: List[str] = Field(default_factory=list)
    grounding_evidence: Dict[str, Any] = Field(default_factory=dict)
    unresolved_hard_constraints: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    requires_clarification: bool = False
    clarification_question: Optional[str] = None
    # Python-computed (never model-filled): True when an unresolved required
    # field is essentially infeasible (unsupported / expired deadline) — the
    # only acceptable outcome is BLOCK, and it outranks requires_clarification.
    requires_block: bool = False
    supersedes: Optional[int] = None
    diff_from_previous: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    def field(self, name: str) -> FieldReading:
        return self.fields.get(name, FieldReading(status="missing"))

    def compact(self) -> Dict[str, Any]:
        """Compact serialisation injected into the background channel."""
        return self.model_dump(mode="json")


def compute_diff(previous: StructuredIntent, current: StructuredIntent) -> Dict[str, Dict[str, Any]]:
    """Deterministic per-field diff between two intent versions."""
    diff: Dict[str, Dict[str, Any]] = {}
    names = set(previous.fields) | set(current.fields)
    for name in sorted(names):
        old = previous.fields.get(name)
        new = current.fields.get(name)
        old_pair = (old.status, old.value) if old else (None, None)
        new_pair = (new.status, new.value) if new else (None, None)
        if old_pair != new_pair:
            diff[name] = {"from": old_pair, "to": new_pair}
    return diff


def _normalize_constraint(text: str) -> set:
    words = "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
    return {w for w in words if len(w) > 2 or w.replace(".", "").isdigit()}


def _dedupe_constraints(intent: StructuredIntent) -> None:
    """A requirement may never sit in BOTH hard_constraints and soft_preferences.
    Direction: hard wins (fail-closed — a possibly-hard constraint must not be
    silently downgraded).  Match is fuzzy (word-set Jaccard >= 0.5) because the
    two listings often differ in surface wording.
    """
    if not intent.hard_constraints or not intent.soft_preferences:
        return
    hard_norm = [_normalize_constraint(h) for h in intent.hard_constraints]
    kept_soft = []
    for soft in intent.soft_preferences:
        soft_norm = _normalize_constraint(soft)
        if not soft_norm:
            kept_soft.append(soft)
            continue
        duplicated = any(
            len(soft_norm & h) / max(1, len(soft_norm | h)) >= 0.5
            for h in hard_norm
        )
        if not duplicated:
            kept_soft.append(soft)
    intent.soft_preferences = kept_soft


def _normalize_span(text: str) -> str:
    """Whitespace-collapsed, case-folded form for containment checks."""
    return " ".join(text.split()).lower()


def _verify_evidence_spans(intent: StructuredIntent, conversation: str) -> None:
    """Deterministic evidence-span containment check (fail-closed).

    Every field/commitment carrying a non-empty evidence span must quote the
    conversation verbatim (modulo whitespace folding and case).  A span that
    does not verify marks the reading ``evidence_verified=False``; a field
    whose status was "explicit" is downgraded to "inferred" — unverified
    evidence may never claim an explicit user statement.
    """
    haystack = _normalize_span(conversation)
    if not haystack:
        return
    for reading in intent.fields.values():
        if reading.evidence and _normalize_span(reading.evidence) not in haystack:
            reading.evidence_verified = False
            if reading.status == "explicit":
                reading.status = "inferred"
    for commitment in intent.commitments:
        if commitment.evidence and _normalize_span(commitment.evidence) not in haystack:
            commitment.evidence_verified = False


def _stamp_confirmations(
    intent: StructuredIntent, previous: Optional[StructuredIntent]
) -> None:
    """User-confirmation stamping: a field that was unresolved in the previous
    version (missing/ambiguous/invalid/unsupported) and is now read as
    explicit/inferred counts as confirmed by the later user turn(s)."""
    unresolved = ("missing", "ambiguous", "invalid", "unsupported")
    for name, reading in intent.fields.items():
        if previous is None:
            reading.confirmed = False
            continue
        old = previous.fields.get(name)
        reading.confirmed = bool(
            old is not None
            and old.status in unresolved
            and reading.status in ("explicit", "inferred")
        )


_ISO_TZ = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\+|-)\d{2}:\d{2}$")


def _parse_iso_tz(value: Any) -> Optional[float]:
    """Parse a timezone-aware ISO timestamp to epoch seconds; None when the
    value is not a string with an explicit offset (or fails to parse)."""
    if not isinstance(value, str) or not _ISO_TZ.match(value):
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _is_expired_deadline(reading: "FieldReading", clock: Optional[str]) -> bool:
    """True when an invalid deadline_at value predates the conversation clock
    (an expired deadline is essentially infeasible — clarification cannot
    repair it). Needs both a parseable value and a parseable clock; anything
    else stays on the RC side."""
    if not clock:
        return False
    deadline_ts = _parse_iso_tz(reading.value)
    clock_ts = _parse_iso_tz(clock)
    return deadline_ts is not None and clock_ts is not None and deadline_ts < clock_ts


def finalize_intent(
    intent: StructuredIntent,
    *,
    catalog_required: Optional[Dict[str, List[str]]] = None,
    previous: Optional[StructuredIntent] = None,
    conversation: str = "",
    conversation_start_at: Optional[str] = None,
) -> StructuredIntent:
    """Deterministic post-processing (fail-closed rules, versioning, diff).

    * evidence verification: when ``conversation`` is given, every non-empty
      evidence span must be contained in it (normalized); unverifiable spans
      mark the reading unverified and downgrade explicit -> inferred;
    * dedupe: the same requirement never appears in both constraint lists
      (hard side kept);
    * commitment levels: when graded ``commitments`` are present, the plain
      hard/soft lists are derived from them (inviolable -> hard, the rest ->
      soft); the dedupe safety net then guarantees no item sits in both;
    * missing_required = fields the scheme requires whose status is not
      explicit/inferred (required set = scheme defaults ∪ live catalog spec);
    * fail-closed split (v4.2): unresolved required fields are graded —
      ``unsupported`` or an expired ``deadline_at`` (invalid value predating
      ``conversation_start_at``) set ``requires_block`` (essentially
      infeasible; BLOCK outranks RC); ``missing``/``ambiguous``/other
      ``invalid`` set ``requires_clarification``;
    * versioning: version = previous.version + 1, supersedes set, diff computed
      in Python (never trusted to the LLM); fields resolved since the previous
      version are stamped confirmed=True.
    """
    if conversation:
        _verify_evidence_spans(intent, conversation)
    if intent.scheme == "purchase" and "payment_method" in intent.fields:
        method = intent.fields["payment_method"]
        if method.status in ("explicit", "inferred"):
            method.value = canonical_payment_method(method.value)
    for commitment in intent.commitments:
        # Only reclassify a plain numeric slippage bound. Keyword presence
        # alone would erase a mandatory MEV route or "skip slippage checks".
        number = r"(?:\d+(?:\.\d+)?|\.\d+)"
        bound = r"(?:must not exceed|do not exceed|cannot exceed|can't exceed|at most|below|under|capped at|cap(?:ped)?(?: of)?|<=|≤)"
        plain_cap = re.fullmatch(
            rf"(?:slippage\s+(?:is\s+)?{bound}\s+{number}\s*(?:%|percent)|"
            rf"{bound}\s+{number}\s*(?:%|percent)\s+slippage)[.!]?",
            commitment.rule.strip(), re.I,
        )
        if commitment.category == "other" and plain_cap:
            commitment.category = "slippage_cap"
        # Narrow fail-closed guard for an explicit instruction to execute at a
        # referenced quote. A past-price observation alone remains descriptive.
        span = commitment.evidence
        explicit_rate = re.search(r"\b(?:execute|buy|sell|swap)\s+at\s+(?:that|this|the quoted)\s+(?:rate|price)\b", span, re.I)
        softened = re.search(r"\b(?:ideally|preferably|hopefully|if possible|around|approximately)\b", span, re.I)
        if (conversation and commitment.category == "other" and commitment.evidence_verified
                and explicit_rate and not softened):
            commitment.level = "inviolable"
            commitment.note = "Explicit execution-at-quote instruction retained; representation unresolved."
    if intent.commitments:
        intent.hard_constraints = [
            c.rule for c in intent.commitments if c.level == "inviolable"
        ]
        intent.soft_preferences = [
            c.rule for c in intent.commitments if c.level != "inviolable"
        ]
    _dedupe_constraints(intent)
    # Lack of an executable representation never changes the user's modality.
    intent.unresolved_hard_constraints = [c.rule for c in intent.commitments
        if c.level == "inviolable" and c.category == "other"]
    required = set(SCHEME_REQUIRED_FIELDS.get(intent.scheme, ("amount",)))
    if catalog_required:
        # the live action catalog is authoritative for executable operations
        for op, params in catalog_required.items():
            if op == intent.scheme:
                required |= set(params)
    missing = []
    block_fields = []
    for name in sorted(required):
        reading = intent.field(name)
        if reading.status not in ("missing", "ambiguous", "invalid",
                                  "unsupported"):
            continue
        missing.append(name)
        # unsupported only blocks when the field is part of the request's own
        # operation shape (operation/asset for every scheme, target_asset for
        # swaps). An unsupported value on an irrelevant optional field (e.g.
        # target_asset on a deposit) is noise, not infeasibility.
        if reading.status == "unsupported" and (
            name in ("operation", "asset")
            or (name == "target_asset" and intent.scheme == "swap")
        ):
            block_fields.append(f"{name} (unsupported)")
        elif (name == "deadline_at" and reading.status == "invalid"
              and _is_expired_deadline(reading, conversation_start_at)):
            block_fields.append(f"{name} (expired deadline)")
    # an expired deadline blocks wherever it was read (required or not): the
    # request is time-infeasible regardless of the scheme's required set.
    # Status is irrelevant — the LLM may legitimately resolve an expired
    # time expression as explicit/inferred with the true (past) value.
    if "deadline_at" not in missing:
        reading = intent.field("deadline_at")
        if _is_expired_deadline(reading, conversation_start_at):
            block_fields.append("deadline_at (expired deadline)")
    intent.missing_required = missing
    intent.requires_block = bool(block_fields)
    if missing:
        intent.requires_clarification = True
        if not intent.clarification_question:
            if intent.requires_block:
                intent.clarification_question = (
                    "Request is not executable: "
                    + ", ".join(block_fields) + "."
                )
            else:
                intent.clarification_question = (
                    "Please specify: " + ", ".join(missing) + "."
                )
    if intent.conflicts:
        intent.requires_clarification = True
        if not intent.clarification_question:
            intent.clarification_question = (
                "Please clarify the unresolved contradiction: "
                + "; ".join(intent.conflicts)
            )
    if intent.unresolved_hard_constraints:
        intent.requires_clarification = True
        intent.clarification_question = "Please clarify how to verify: " + "; ".join(intent.unresolved_hard_constraints)

    if previous is not None:
        intent.version = previous.version + 1
        intent.supersedes = previous.version
        intent.diff_from_previous = compute_diff(previous, intent)
    else:
        intent.version = 1
        intent.supersedes = None
        intent.diff_from_previous = {}
    _stamp_confirmations(intent, previous)
    return intent
