"""H1/H2/H3 ablation hook protocols.

The baseline (B000) wires NoOp implementations at three fixed call points so
that later method modules plug in without touching the skeleton:

  * H1 结构化意图 -> ``IntentStructurer``: called once per case by the runner
    after the visible conversation is fixed and before the team is built.
    Its product is injected into the background (B) channel.  B000: returns
    the raw conversation unchanged.

  * H2 意图-交易对齐验证 -> ``AlignmentVerifier``: called by the runner right
    after the Evaluation team returns to the Main Supervisor, with the
    current intermediate output (proposal/final_decision) and the ledger
    receipts.  Its evidence would travel the M/B/I channels.  B000: returns
    None (no independent evidence).

  * H3 动态边界与 HITL -> ``DecisionPolicy``: called by the runner when the
    Main Supervisor produced a legal FINISH, with the final decision and the
    ledger snapshot.  It may adjust the decision (execute/wait/clarify/
    escalate).  B000: returns the decision unchanged.

Any flag other than all-off raises NotImplementedError in the baseline build;
this keeps B000 the only executable condition until the H modules land.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


class IntentStructurer(Protocol):
    def configure(self, **case_context) -> None: ...

    def structure(self, visible_conversation: str) -> Any: ...

    def on_user_event(
        self, user_events: List[str], visible_conversation: str
    ) -> Optional[Any]: ...


class AlignmentVerifier(Protocol):
    def configure(self, **case_context) -> None: ...

    def verify(
        self,
        intermediate_output: Any,
        receipts: List[Dict[str, Any]],
        environment_snapshot: Dict[str, Any],
        intent: Any = None,
    ) -> Optional[Dict[str, Any]]: ...


class DecisionPolicy(Protocol):
    def configure(self, **case_context) -> None: ...

    def on_environment_event(
        self,
        new_events: List[str],
        previous_facts: Dict[str, Any],
        current_facts: Dict[str, Any],
        intent: Any = None,
    ) -> Optional[str]: ...

    def decide(
        self, final_decision: Dict[str, Any], environment_snapshot: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]: ...


class NoOpIntentStructurer:
    def configure(self, **case_context) -> None:
        return None

    def structure(self, visible_conversation: str) -> Any:
        return visible_conversation

    def on_user_event(self, user_events, visible_conversation):
        return None


class NoOpAlignmentVerifier:
    def configure(self, **case_context) -> None:
        return None

    def update_conversation(self, conversation: str) -> None:
        return None

    def verify(self, intermediate_output, receipts, environment_snapshot, intent=None):
        return None


class NoOpDecisionPolicy:
    def configure(self, **case_context) -> None:
        return None

    def on_environment_event(self, new_events, previous_facts, current_facts,
                             intent=None):
        return None

    def decide(self, final_decision, environment_snapshot, **kwargs):
        return final_decision


@dataclass(frozen=True)
class HookSet:
    h1: IntentStructurer
    h2: AlignmentVerifier
    h3: DecisionPolicy


def condition_name(h1: bool, h2: bool, h3: bool) -> str:
    return "B" + ("1" if h1 else "0") + ("1" if h2 else "0") + ("1" if h3 else "0")


def build_hooks(h1: bool = False, h2: bool = False, h3: bool = False, llm=None) -> HookSet:
    """Wire hooks for an ablation condition. H1/H2/H3 are all implemented
    (B000–B111 executable). Build hooks per case (modules carry per-case
    state). H3's deterministic policy needs no LLM; H1/H2 do."""
    if (h1 or h2) and llm is None:
        raise ValueError("H1/H2 require an LLM for their module components")
    if h1:
        from payment.h1.structurer import H1IntentStructurer

        h1_obj: IntentStructurer = H1IntentStructurer(llm)
    else:
        h1_obj = NoOpIntentStructurer()
    if h2:
        from payment.h2.verifier import H2AlignmentVerifier

        h2_obj: AlignmentVerifier = H2AlignmentVerifier(llm)
    else:
        h2_obj = NoOpAlignmentVerifier()
    if h3:
        from payment.h3.policy import H3DecisionPolicy

        h3_obj: DecisionPolicy = H3DecisionPolicy()
    else:
        h3_obj = NoOpDecisionPolicy()
    return HookSet(h1=h1_obj, h2=h2_obj, h3=h3_obj)
