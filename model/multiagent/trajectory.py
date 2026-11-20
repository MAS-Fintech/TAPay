"""Canonical agent-trajectory recording and normalization.

HF1/ASR are computed from the *semantic handoff* view of a run: pure
intra-team dispatch hops (member -> supervisor -> member of the same team)
are collapsed, while team-boundary crossings, nested-team returns and the
final FINISH turn are preserved.  This module records raw node-entry events
(one per actual agent execution) and normalizes them into the canonical
trajectory whose node names match the dataset's workflow_reference strings
verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple


@dataclass(frozen=True)
class TeamLayout:
    """Static hierarchy description used by the normalization rule.

    supervisors:   canonical supervisor name -> canonical member names
    leaf_members:  members that are plain ReactAgents (nested-team
                   supervisors are members of their parent team but NOT leafs)
    """

    supervisors: Dict[str, FrozenSet[str]]
    leaf_members: FrozenSet[str]

    def is_member_of(self, node: str, supervisor: str) -> bool:
        return node in self.supervisors.get(supervisor, frozenset())

    def is_leaf_member_of(self, node: str, supervisor: str) -> bool:
        return self.is_member_of(node, supervisor) and node in self.leaf_members


class TrajectoryRecorder:
    """Per-run sink for raw agent node-entry events (execution order)."""

    def __init__(self) -> None:
        self.entries: List[str] = []

    def enter(self, node: str) -> None:
        self.entries.append(node)


def normalize_trajectory(raw: List[str], layout: TeamLayout) -> List[str]:
    """Collapse pure intra-team dispatch hops from a raw node-entry sequence.

    A supervisor entry S is removed iff the previously kept node is a *leaf*
    member of S's own team AND S's next hop is also a member of S's own team.
    Everything else (team entries from a parent, returns to a parent, the
    final FINISH turn, entries after a nested-team supervisor) is kept.
    """
    kept: List[str] = []
    n = len(raw)
    for i, node in enumerate(raw):
        if node in layout.supervisors and kept:
            prev_kept = kept[-1]
            nxt = raw[i + 1] if i + 1 < n else None
            if (
                layout.is_leaf_member_of(prev_kept, node)
                and nxt is not None
                and layout.is_member_of(nxt, node)
            ):
                continue
        kept.append(node)
    return kept


def handoff_edges(trajectory: List[str]) -> List[Tuple[str, str]]:
    """Unique directed handoff edges, first-occurrence order (HF1 view)."""
    seen = set()
    edges: List[Tuple[str, str]] = []
    for a, b in zip(trajectory, trajectory[1:]):
        if (a, b) not in seen:
            seen.add((a, b))
            edges.append((a, b))
    return edges


def transitions(trajectory: List[str]) -> List[Tuple[str, str]]:
    """Ordered adjacent transition multiset (ASR view); repeats kept."""
    return list(zip(trajectory, trajectory[1:]))
