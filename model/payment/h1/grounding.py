"""Resolve purchase identity only from product records already returned by tools.

No access to the hidden catalog, proposal values, simulation outcomes or gold.
Exact identifiers are intentional: fuzzy matching would erase P4 substitutions.
"""
import re
from payment.h1.schema import FieldReading


def _contains(text, value):
    if not isinstance(value, str) or not value.strip():
        return False
    return re.search(r"(?<!\w)" + re.escape(value.strip()) + r"(?!\w)",
                     text, flags=re.IGNORECASE) is not None


def defer_sku(intent):
    if intent.scheme != "purchase":
        return intent
    # A display name in the SKU field still needs tool grounding. Preserve the
    # existing path for compact identifiers; simulator/catalog checks remain.
    reading = intent.field("sku")
    display_name = isinstance(reading.value, str) and bool(re.search(r"\s", reading.value))
    if "sku" not in intent.missing_required and not display_name:
        return intent
    if reading.status not in ("missing", "inferred", "explicit"):
        return intent
    intent.pending_resolution = list(dict.fromkeys([*intent.pending_resolution, "sku"]))
    intent.missing_required = [f for f in intent.missing_required if f != "sku"]
    if not intent.missing_required and not intent.conflicts and not intent.unresolved_hard_constraints:
        intent.requires_clarification = False
        intent.clarification_question = None
    return intent


def _resolve_identity_commitments(intent, product, source):
    """Recognize only complete, exact-variant statements, never keyword hits.

    Keep a mandatory route, seller, delivery term etc. unresolved even if it
    mentions this model. A successful purchase is never identity evidence.
    """
    names = [product.get('sku'), product.get('model'), product.get('title')]
    if product.get('brand') and product.get('model'):
        names.append(product['brand'] + ' ' + product['model'])
    names = [re.escape(n) for n in names if n]
    identity = '(?:' + '|'.join(names) + ')'
    for c in intent.commitments:
        if c.category != 'other' or not c.evidence_verified or not c.evidence:
            continue
        if ' '.join(c.evidence.split()).casefold() not in ' '.join(source.split()).casefold():
            continue
        pattern = (rf'(?:purchase|buy|order) (?:exactly )?(?:(one) )?(?:the )?{identity}'
                   rf'(?: variant)?(?: specifically)?,? not (?:any other|another|other) variant[.!]?')
        match = re.fullmatch(pattern, c.rule.strip(), re.I)
        if match and (not match.group(1) or intent.field('quantity').value == 1):
            c.category = 'product_identity'
            c.note = 'Exact variant verified by observed shop tool identity; strength unchanged.'
    intent.unresolved_hard_constraints = [c.rule for c in intent.commitments
        if c.level == 'inviolable' and c.category == 'other']


def ground_sku(intent, observations, conversation, state_version):
    if intent.scheme != "purchase" or not ("sku" in intent.pending_resolution or
                                           "sku" in intent.missing_required):
        return None
    observed = {}
    observed_all = {}
    for item in observations:
        for p in item.get("products", []):
            # A product observed before the revision is still real catalog
            # identity: "not the CM-1" is an exclusion of THAT product, not a
            # complex negation, even when CM-1 was seen at an earlier state.
            observed_all[p["sku"]] = p
            if item.get("state_version") == state_version:
                observed[p["sku"]] = p
    if not observed:
        return None
    # Last user turn that names a product wins; quantity-only revisions carry it.
    turns = re.findall(r"(?:^|\n)user:\s*(.*?)(?=\n(?:user|assistant|environment|system):|\Z)",
                       conversation, flags=re.IGNORECASE | re.DOTALL)
    if not turns and not re.search(r"(?:^|\n)\w+:", conversation):
        turns = [conversation]
    matches = []
    source = ""
    for turn in reversed(turns):
        corrections = list(re.finditer(r"\b(?:actually|rather(?!\s+than)|I mean)\b", turn, re.I))
        selection = turn[corrections[-1].end():] if corrections else turn
        # A trailing exclusion must not erase the positive choice before it.
        parts = re.split(r",\s*(?:instead,?\s*)?not\s+(?:the\s+)?", selection, maxsplit=1, flags=re.I)
        if len(parts) == 2 and any(_contains(parts[1], p.get('sku')) or
                                  _contains(parts[1], p.get('model')) for p in observed_all.values()):
            selection = parts[0]
        if re.search(r"\b(?:not|don't|do not)\s+(?:buy|order|purchase|choose|select|want|the)\b", selection, re.I):
            # Negated identities are not positive selections. Leave complex
            # negation to clarification rather than accidentally resurrecting it.
            return None
        for product in observed.values():
            for identifier in (product.get('sku'), product.get('model'), product.get('brand')):
                if identifier and re.search(r"\b(?:not|except|avoid)\s+(?:the\s+)?" +
                                             re.escape(identifier) + r"(?!\w)", selection, re.I):
                    return None
        hits = []
        for p in observed.values():
            identifiers = [p.get('sku')]
            if _contains(turn, p.get('brand')):
                identifiers.append(p.get('model'))
                # Exact multiword suffixes allow "SE Pro" after "LF-SE" was
                # explicitly corrected. No edit distance or homoglyph folding.
                model = p.get('model') or ''
                suffix = model.split('-', 1)[-1]
                if corrections and len(suffix.split()) >= 2:
                    identifiers.append(suffix)
            for value in identifiers:
                if not isinstance(value, str) or not value.strip():
                    continue
                hits.extend((m.start(), m.end(), p) for m in re.finditer(
                    r'(?<!\w)' + re.escape(value.strip()) + r'(?!\w)', selection, re.I))
        # Suppress a base-model prefix only at the SAME textual mention. Two
        # distinct candidate mentions remain ambiguous regardless of length.
        maximal = [(start,end,p) for start,end,p in hits if not any(
            other_start <= start and end <= other_end and (other_start,other_end)!=(start,end)
            for other_start,other_end,_ in hits)]
        matches = list({p['sku']:p for _,_,p in maximal}.values())
        if matches:
            source = turn.strip()
            break
        if corrections:
            # Only explicitly non-identity revisions may carry the earlier product.
            # Unknown product replacements still stop; never fall back to old SKU.
            non_identity = re.search(r"\b(?:cap|budget|quantity|payment|credit card|debit card|paypal)\b", turn, re.I)
            product_change = re.search(r"\b(?:buy|order|purchase|get|model|product|variant|different|make it)\b|[A-Za-z]+-\d", turn, re.I)
            if not non_identity or product_change:
                return None
    if len(matches) != 1:
        # Tool evidence did not establish a unique requested identity. Never use
        # the model's proposed/simulated SKU to resolve this circularly.
        return None
    updated = intent.model_copy(deep=True)
    p = matches[0]
    updated.fields["sku"] = FieldReading(value=p["sku"], status="inferred",
        evidence=source, hardness="hard", evidence_verified=True)
    updated.pending_resolution = [f for f in updated.pending_resolution if f != "sku"]
    updated.missing_required = [f for f in updated.missing_required if f != "sku"]
    updated.grounding_evidence["sku"] = {"source": "observed_shop_tool",
        "state_version": state_version, "sku": p["sku"], "brand": p.get("brand"),
        "model": p.get("model"), "conversation_span": source}
    _resolve_identity_commitments(updated, p, source)
    updated.open_or_ambiguous = any(r.status == 'ambiguous' for r in updated.fields.values())
    if (not updated.missing_required and not updated.conflicts and not updated.unresolved_hard_constraints
            and not updated.open_or_ambiguous):
        updated.requires_clarification = False
        updated.clarification_question = None
    return updated


# A first-person completion claim: the user states they ALREADY signed/approved
# a fresh authorization ("I already approved a fresh one", "just signed").
# Procedural demands ("don't execute until I approve") carry no such claim and
# must stay unresolved regardless of execution evidence.
_COMPLETION_CLAIM_RE = re.compile(
    r"\b(?:already|just)\b[^.!]*\b(?:approv\w*|authori[sz]\w*|sign\w*|allowance)\b|"
    r"\b(?:approv\w*|authori[sz]\w*|sign\w*|allowance)\b[^.!]*\b(?:already|just)\b",
    re.IGNORECASE,
)

# The commitment itself must concern an environment-held authorization, not
# any other signing act ("I already signed the paper form" is not this).
_AUTHORIZATION_RULE_RE = re.compile(
    r"\b(?:approv\w*|authori[sz]\w*|allowance)\b", re.IGNORECASE)


def resolve_authorization_commitments(intent, receipts, state_version):
    """Satisfy 'fresh approval already signed' commitments from execution evidence.

    The structurer is blind to account facts (visibility contract), so an
    inviolable approval commitment stays unresolved even when the environment
    already holds the fresh authorization. A successful receipt bound to the
    CURRENT state is team-acquired proof that the transaction runs under the
    environment's current authorization — the catalog exposes no approval
    parameter, so executing on the current state is the only way to honor
    "use the fresh approval". The user's own completion claim must be present
    in the verified evidence; without it the commitment is procedural and
    stays unresolved. Fail-closed both ways: no completion claim -> None, no
    current-state approval-bearing receipt -> None (e.g. the environment shows
    no active approval, so simulation never succeeds). Strength unchanged on
    reclassification; mirrors ground_sku's evidence-only rule.
    """
    targets = [
        c for c in intent.commitments
        if c.level == "inviolable" and c.category == "other"
        and c.evidence_verified and c.evidence
        and _AUTHORIZATION_RULE_RE.search(c.rule + " " + c.evidence)
        and _COMPLETION_CLAIM_RE.search(c.evidence)
    ]
    if not targets:
        return None
    proof = next(
        (r for r in receipts
         if r.get("status") == "SIMULATED_EXECUTED"
         and r.get("state_version") == state_version
         and r.get("approval_present")),
        None,
    )
    if proof is None:
        return None
    rules = {c.rule for c in targets}
    updated = intent.model_copy(deep=True)
    changed = False
    for c in updated.commitments:
        if c.rule in rules and c.level == "inviolable" and c.category == "other":
            c.category = "authorization"
            c.note = ("Fresh authorization confirmed by receipt "
                      f"{proof.get('receipt_id')} bound to the current state; "
                      "strength unchanged.")
            changed = True
    if not changed:
        return None
    updated.unresolved_hard_constraints = [
        c.rule for c in updated.commitments
        if c.level == "inviolable" and c.category == "other"
    ]
    updated.grounding_evidence["authorization"] = {
        "source": "observed_simulation_receipt",
        "state_version": state_version,
        "receipt_id": proof.get("receipt_id"),
        "authorization_version": proof.get("authorization_version"),
    }
    if (not updated.missing_required and not updated.conflicts
            and not updated.unresolved_hard_constraints
            and not updated.open_or_ambiguous):
        updated.requires_clarification = False
        updated.clarification_question = None
    return updated
