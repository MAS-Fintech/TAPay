"""Control-agent hook protocols for TAPay component ablations.

The backbone-only condition uses NoOp implementations. Enabled components
provide structured intent, transaction-intent alignment and dynamic decision
boundaries through the same interfaces. The runner owns hook timing,
revision handling and final enforcement. Internal h1/h2/h3 names are retained
for compatibility with existing experiment configuration identifiers.
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


_H3_POLICY_OVERRIDE_KEYS = {"risk_threshold", "risk_enter", "risk_exit",
                            "min_hold_events", "pressure_threshold",
                            "clarify_budget"}


def build_hooks(h1: bool = False, h2: bool = False, h3: bool = False, llm=None,
                h3_boundary: Optional[str] = None,
                h3_features: Optional[set] = None,
                h3_policy_overrides: Optional[Dict[str, Any]] = None) -> HookSet:
    """Wire hooks for an ablation condition. H1/H2/H3 are all implemented
    (B000–B111 executable). Build hooks per case (modules carry per-case
    state). H3's deterministic policy needs no LLM; H1/H2 do.

    ``h3_boundary`` optionally points at a learned boundary artifact JSON
    (see scripts/h3/learn_boundary.py); when omitted the H3 policy keeps its
    default fixed-threshold behaviour exactly. ``h3_features`` names the
    optional H3 sub-features (slow_loop / feedback_replan /
    clarify_policy); default None = all off, behaviour unchanged.
    ``h3_policy_overrides`` passes whitelisted constructor kwargs
    (risk_enter/risk_exit/min_hold_events/pressure_threshold/...) to the H3
    policy for RQ4 sensitivity sweeps; it cannot combine with a boundary
    artifact and defaults to no behaviour change."""
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

        if h3_policy_overrides:
            if h3_boundary:
                raise ValueError(
                    "h3_policy_overrides cannot combine with a learned "
                    "boundary artifact; pick one parameter source")
            unknown = set(h3_policy_overrides) - _H3_POLICY_OVERRIDE_KEYS
            if unknown:
                raise ValueError(
                    f"unknown H3 policy override keys: {sorted(unknown)}")
        h3_obj: DecisionPolicy = (
            H3DecisionPolicy.from_boundary_file(h3_boundary,
                                                features=h3_features)
            if h3_boundary else H3DecisionPolicy(features=h3_features,
                                                 **(h3_policy_overrides or {}))
        )
    else:
        h3_obj = NoOpDecisionPolicy()
    return HookSet(h1=h1_obj, h2=h2_obj, h3=h3_obj)
