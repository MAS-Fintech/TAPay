"""RQ3 fault injection: field-level distortion of the team's final proposal.

Experiment-only harness.  A spec names one parameter and a mode:

  "amount:*2"       multiply the numeric value by a factor
  "amount:+20%"     relative increase by a percentage
  "amount:=20000"   set an absolute value

The runner applies it to the team's final EXECUTE proposal right before the
final H2 re-validation and the H3 decide, then persists the injection record
so an injected distortion is distinguishable from a model-spontaneous error.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def parse_fault_spec(text: str) -> Dict[str, Any]:
    if ":" not in text:
        raise ValueError(f"fault spec needs 'parameter:mode', got {text!r}")
    param, mode = text.split(":", 1)
    param = param.strip()
    mode = mode.strip()
    if not param:
        raise ValueError("fault spec has empty parameter")
    if mode.startswith("*"):
        return {"parameter": param, "mode": "multiply", "factor": float(mode[1:])}
    if mode.startswith("+"):
        return {"parameter": param, "mode": "increase_pct",
                "pct": float(mode[1:].rstrip("%"))}
    if mode.startswith("="):
        raw = mode[1:]
        try:
            value: Any = float(raw)
        except ValueError:
            value = raw  # String-valued injection, e.g. asset:=BALX substitution.
        return {"parameter": param, "mode": "set", "value": value}
    raise ValueError(f"unknown fault mode in {text!r} (use *k, +p%, =v)")


def apply_fault_injection(
    parameters: Optional[Dict[str, Any]],
    spec: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    params = dict(parameters or {})
    param = spec["parameter"]
    before = params.get(param)
    record: Dict[str, Any] = {"parameter": param, "before": before, "applied": False}
    if before is None:
        record["note"] = "parameter absent; no injection"
        return params, record
    if spec["mode"] == "set":
        after = spec["value"]
    elif not isinstance(before, (int, float)) or isinstance(before, bool):
        record["note"] = "parameter not numeric; no injection"
        return params, record
    elif spec["mode"] == "multiply":
        after = before * spec["factor"]
    else:
        after = before * (1.0 + spec["pct"] / 100.0)
    params[param] = after
    record.update({"after": after, "applied": True, "mode": spec["mode"]})
    return params, record
