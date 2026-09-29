"""Final structured prediction contract and run failure categories.

The only accepted prediction source is the Main Supervisor's structured
``intermediate_output`` at a legal FINISH.  Failures are classified, never
rewritten into a semantic label: a crashed run that happens to look like
BLOCK must not earn a correct classification (RQ2 spec).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

BEHAVIORS = ("EXECUTE", "BLOCK", "REQUEST_CLARIFICATION")

FAILURE_CATEGORIES = (
    "PROVIDER_FAILURE",
    "PARSE_FAILURE",
    "TOOL_FAILURE",
    "ROUND_LIMIT",
    "INCOMPLETE",
)

# Keys searched (in order) inside the final intermediate_output object.
FINAL_DECISION_KEYS = ("final_decision", "final_output", "decision")


@dataclass
class Prediction:
    final_behavior: Optional[str] = None
    action_id: Optional[str] = None
    operation: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    receipt_id: Optional[str] = None
    rationale: Optional[str] = None
    receipt: Optional[Dict[str, Any]] = None  # resolved from the ledger
    legal_finish: bool = False
    failure: Optional[str] = None  # one of FAILURE_CATEGORIES
    failure_detail: str = ""
    raw_final_output: Any = None

    @property
    def is_valid(self) -> bool:
        return self.failure is None and self.final_behavior in BEHAVIORS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_behavior": self.final_behavior,
            "action_id": self.action_id,
            "operation": self.operation,
            "parameters": self.parameters,
            "receipt_id": self.receipt_id,
            "rationale": self.rationale,
            "receipt": self.receipt,
            "legal_finish": self.legal_finish,
            "failure": self.failure,
            "failure_detail": self.failure_detail,
            "raw_final_output": self.raw_final_output,
        }


def _first_present(obj: Dict[str, Any], keys) -> Optional[Any]:
    for k in keys:
        if isinstance(obj, dict) and k in obj and obj[k] is not None:
            return obj[k]
    return None


def extract_prediction(final_output: Any, ledger) -> Prediction:
    """Extract the final decision from a legal-FINISH intermediate output.

    ``ledger`` is the case's EnvironmentLedger, used only to resolve the
    receipt the model cited (never to synthesize or amend the behavior).
    """
    pred = Prediction(raw_final_output=final_output, legal_finish=True)
    if not isinstance(final_output, dict):
        pred.failure = "PARSE_FAILURE"
        pred.failure_detail = "final output is not a JSON object"
        return pred

    decision = _first_present(final_output, FINAL_DECISION_KEYS)
    if isinstance(decision, dict):
        source = decision
    elif final_output.get("final_behavior") in BEHAVIORS:
        source = final_output  # tolerate flat layouts
    else:
        # Adaptive routes without a Revisor step end with the Evaluation
        # team's output: the decision then comes from the Generator's
        # confirmed "decision_recommendation" plus the preserved "proposal".
        proposal = final_output.get("proposal")
        proposal = proposal if isinstance(proposal, dict) else {}
        recommendation = final_output.get("decision_recommendation") or proposal.get(
            "decision_recommendation"
        )
        source = {
            "final_behavior": recommendation,
            "action_id": proposal.get("action_id"),
            "operation": proposal.get("operation"),
            "parameters": proposal.get("parameters"),
            "receipt_id": proposal.get("receipt_id"),
            "rationale": final_output.get("rationale"),
        }
    decision = source

    behavior = decision.get("final_behavior") or final_output.get("final_behavior")
    if isinstance(behavior, str):
        behavior = behavior.strip().upper()
    if behavior not in BEHAVIORS:
        pred.failure = "PARSE_FAILURE"
        pred.failure_detail = f"missing/invalid final_behavior: {behavior!r}"
        return pred

    params = decision.get("parameters")
    if params is None:
        params = final_output.get("parameters")
    if params is not None and not isinstance(params, dict):
        pred.failure = "PARSE_FAILURE"
        pred.failure_detail = "parameters is not a JSON object"
        return pred

    pred.final_behavior = behavior
    pred.action_id = decision.get("action_id") or final_output.get("action_id")
    pred.operation = decision.get("operation") or final_output.get("operation")
    pred.parameters = params
    pred.receipt_id = decision.get("receipt_id") or final_output.get("receipt_id")
    pred.rationale = decision.get("rationale") or final_output.get("rationale")

    # Intent-level timing/budget fields may legitimately live at the decision
    # level (they are not action-catalog parameters); promote them into the
    # merged parameter view used by the task-contract check.
    for key in ("time_window_seconds", "deadline_at", "max_total_usd"):
        if (pred.parameters is None or key not in pred.parameters) and decision.get(
            key
        ) is not None:
            if pred.parameters is None:
                pred.parameters = {}
            pred.parameters[key] = decision[key]

    if pred.receipt_id:
        for receipt in ledger.receipts:
            if receipt.get("receipt_id") == pred.receipt_id:
                pred.receipt = receipt
                break
    return pred


def failure_prediction(category: str, detail: str) -> Prediction:
    if category not in FAILURE_CATEGORIES:
        category = "INCOMPLETE"
    return Prediction(failure=category, failure_detail=detail, legal_finish=False)
