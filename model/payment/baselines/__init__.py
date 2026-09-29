"""Literature-baseline registry with auto-discovery.

Each literature baseline lives in its own module inside this package and
exposes a module-level ``BASELINE: BaselineSpec``.  Importing this package
scans and registers every sibling module, so adding a new baseline never
requires editing shared files.

``b000`` (the TalkHier team runner) is the built-in default path and is
intentionally NOT registered here; only non-B000 literature baselines live in
this registry.  See docs/RQ1_BASELINES.md for the interface contract.
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from multiagent.trajectory import TeamLayout


@dataclass(frozen=True)
class BaselineSpec:
    """One literature baseline's wiring.

    name:               CLI name (``--baselines <name>``), lowercase.
    workflow_key:       key used to look up the reference workflow:
                        ``case.workflow_reference[workflow_key]`` first, then
                        ``data/workflow_refs/<workflow_key>.json``.
    team_layout:        hierarchy for trajectory normalization; single-agent
                        baselines use an identity (empty) layout.
    run_case:           ``(case, llm, hooks, *, release_policy,
                        recursion_limit, verbose) -> CaseResult``.
    build_replay_model: scripted model for ``--mock`` offline replay
                        (validation only); None means no mock support.
    """

    name: str
    workflow_key: str
    team_layout: Callable[[], TeamLayout]
    run_case: Callable[..., Any]
    build_replay_model: Optional[Callable[..., Any]] = None


_REGISTRY: Dict[str, BaselineSpec] = {}


def register(spec: BaselineSpec) -> None:
    if spec.name in _REGISTRY:
        raise ValueError(f"duplicate baseline registration: {spec.name!r}")
    if spec.name == "b000":
        raise ValueError("b000 is the built-in default path; it is not a "
                         "registry baseline")
    _REGISTRY[spec.name] = spec


def get_baseline(name: str) -> BaselineSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown baseline {name!r}; available: {available_baselines()}"
        ) from None


def available_baselines() -> List[str]:
    return sorted(_REGISTRY)


def _discover() -> None:
    for mod in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{__name__}.{mod.name}")
        spec = getattr(module, "BASELINE", None)
        if spec is not None:
            register(spec)


_discover()
