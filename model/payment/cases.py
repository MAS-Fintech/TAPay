"""Payment case loading and visibility partitioning.

A merged case has six parts (see data spec): ``input``, ``runtime``,
``evaluation.gold``, ``evaluation.task_reference``,
``evaluation.workflow_reference`` and ``review``.

Visibility contract (enforced here and in the runner):
  * model-visible: ``input.conversation`` visible turns (dynamic successor
    turns stay hidden until the environment ledger releases them), plus
    whatever the environment/simulator tools return;
  * runner-owned only: ``runtime.environment`` (queried through restricted
    tools) and ``runtime.execution_policy`` resource/event policy. Historical
    inputs without execution_policy retain task_reference budgets only for
    explicitly requested historical comparisons;
  * scorer-only: ``evaluation.gold``, the rest of ``task_reference``,
    ``workflow_reference``; ``review`` never enters the pipeline;
  * ``case_id`` / ``split`` / provenance labels never enter the model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

REQUIRED_TOP_KEYS = ("case_id", "dataset_kind", "input", "runtime", "evaluation")


@dataclass
class PaymentCase:
    case_id: str
    dataset_kind: str  # "static" | "dynamic"
    conversation: List[Dict]  # all turns, including initially hidden ones
    environment: Dict  # runtime.environment (runner-owned)
    gold: Dict
    task_reference: Dict
    workflow_reference: Optional[Dict]
    split: str
    review: Dict = field(default_factory=dict)

    runtime_policy: Dict = field(default_factory=dict)

    @property
    def simulation_policy(self):
        if self.runtime_policy:
            return self.runtime_policy
        return self.task_reference  # historical input compatibility only

    # -- model-visible conversation ----------------------------------------
    @property
    def successor_visible_after_turn_id(self) -> Optional[int]:
        if self.environment.get("successor_state") is None:
            return None
        return self.environment.get("successor_visible_after_turn_id")

    def visible_turns(self) -> List[Dict]:
        """Turns visible at run start: successor-gated turns stay hidden.
        Turns appended by the simulated-user diagnostic (marked
        ``origin="simulated_user"``) are always visible."""
        gate = self.successor_visible_after_turn_id
        if gate is None:
            return list(self.conversation)
        return [
            t
            for t in self.conversation
            if t.get("turn_id", 0) < gate or t.get("origin") == "simulated_user"
        ]

    def hidden_turns(self) -> List[Dict]:
        gate = self.successor_visible_after_turn_id
        if gate is None:
            return []
        return [
            t
            for t in self.conversation
            if t.get("turn_id", 0) >= gate and t.get("origin") != "simulated_user"
        ]

    def render_visible_conversation(self) -> str:
        lines = []
        for turn in self.visible_turns():
            role = turn.get("role", "user")
            lines.append(f"{role}: {turn.get('content', '')}")
        return "\n".join(lines)


def load_cases(path: str) -> List[PaymentCase]:
    cases: List[PaymentCase] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            missing = [k for k in REQUIRED_TOP_KEYS if k not in raw]
            if missing:
                raise ValueError(f"case line {lineno} missing keys: {missing}")
            evaluation = raw["evaluation"]
            policy = raw["runtime"].get("execution_policy", {})
            if policy:
                if (policy.get("protocol") != "round6-v1"
                        or policy.get("simulation_permitted") is not True
                        or type(policy.get("simulation_max_calls")) is not int
                        or policy["simulation_max_calls"] < 1
                        or policy.get("simulation_per_proposal_max_calls") != 1
                        or policy.get("event_completion") != "before_finish"
                        or policy.get("purchase_price_consent") != "explicit_total_cap"):
                    raise ValueError("Invalid round6 runtime execution policy")
            cases.append(
                PaymentCase(
                    case_id=raw["case_id"],
                    dataset_kind=raw.get("dataset_kind", "static"),
                    conversation=raw["input"]["conversation"],
                    environment=raw["runtime"]["environment"],
                    gold=evaluation.get("gold", {}),
                    task_reference=evaluation.get("task_reference", {}),
                    workflow_reference=evaluation.get("workflow_reference"),
                    split=raw.get("split", ""),
                    review=raw.get("review", {}),
                    runtime_policy=raw["runtime"].get("execution_policy", {}),
                )
            )
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case_id in input; outputs require unique case IDs")
    return cases
