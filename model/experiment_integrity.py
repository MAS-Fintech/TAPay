"""Frozen run identities and append-only attempts shared by both CLIs."""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def select_cases(cases, split, require_reviewed=False):
    selected = [c for c in cases if split == "all" or c.split == split]
    ids = [c.case_id for c in selected]
    if not ids:
        raise ValueError("No cases in selected split; use --split all for legacy/diagnostic input.")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case_id would alias output paths; assign unique IDs first.")
    if any(Path(i).name != i or i in (".", "..") or "/" in i or "\\" in i for i in ids):
        raise ValueError("Unsafe case_id path")
    if require_reviewed and any(c.review.get("formal_acc_eligible") is not True for c in selected):
        raise ValueError("Selected cases include unreviewed/ineligible references; complete review before formal evaluation")
    return selected


def freeze_run(root, identity):
    """Validate before provider construction or execution; never relabel old runs."""
    root = Path(root)
    p = root / "run_identity.json"
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        if old != identity:
            changed = sorted(k for k in old.keys() | identity.keys() if old.get(k) != identity.get(k))
            raise ValueError("Resume identity mismatch: " + ", ".join(changed))
    elif any(root.iterdir()):
        raise ValueError("Unversioned existing output cannot be resumed; use a fresh output directory.")
    else:
        write_json(p, identity)
    return digest(identity)


def run_identity(cases, config, code_hashes, **settings):
    # Credentials are neither stored nor hashed; operational model settings are.
    cfg = {s: {k: v for k, v in config.items(s)
               if not any(token in k.lower() for token in ("key", "secret", "token", "password"))}
           for s in config.sections()}
    return {"protocol": "audit-v1", "cases_sha256": digest([asdict(c) for c in cases]),
            "case_ids": [c.case_id for c in cases], "config": cfg,
            "code_sha256": code_hashes, **settings}


def failure_record(case_id, detail, wall_seconds=0.0):
    return {"case_id": case_id, "final_behavior": None, "failure": "INCOMPLETE",
            "failure_detail": detail, "legal_finish": False, "parameters": {},
            "agent_trajectory": [], "handoff_sequence": [], "receipts": [],
            "tool_log": [], "released_events": [], "final_state_version": 1,
            "sim_calls": 0, "wall_seconds": wall_seconds}


def retryable(record):
    return record.get("failure") == "PROVIDER_FAILURE" or (
        record.get("failure") == "INCOMPLETE" and
        record.get("failure_detail", "").startswith(("case_timeout:", "interrupted_attempt:")))


class AttemptStore:
    def __init__(self, root, identity_hash, retry_infrastructure=False, max_attempts=1):
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.root = Path(root)
        self.identity_hash = identity_hash
        self.retry_infrastructure = retry_infrastructure
        self.max_attempts = max_attempts
        self.lock = threading.Lock()

    def case_dir(self, condition, case_id):
        return self.root / condition / "cases" / case_id if condition else self.root / "cases" / case_id

    def begin(self, condition, case_id):
        case_dir = self.case_dir(condition, case_id)
        attempts = sorted((case_dir / "attempts").glob("[0-9]*"))
        # A killed process may leave a started attempt but no result. Preserve it.
        if attempts and not (attempts[-1] / "prediction.json").exists():
            self.finish(attempts[-1], failure_record(case_id, "interrupted_attempt: no terminal record"))
        # Recover a process killed between writing the durable attempt and
        # publishing its compatibility view at cases/<id>/prediction.json.
        if attempts and (attempts[-1] / "prediction.json").exists():
            last = json.loads((attempts[-1] / "prediction.json").read_text(encoding="utf-8"))
            write_json(case_dir / "prediction.json", last)
            trajectory = attempts[-1] / "trajectory.jsonl"
            if trajectory.exists():
                (case_dir / "trajectory.jsonl").write_bytes(trajectory.read_bytes())
        current = case_dir / "prediction.json"
        if current.exists():
            rec = json.loads(current.read_text(encoding="utf-8"))
            if not (self.retry_infrastructure and retryable(rec) and len(attempts) < self.max_attempts):
                return None
        if len(attempts) >= self.max_attempts:
            return None
        attempt = case_dir / "attempts" / f"{len(attempts) + 1:04d}"
        attempt.mkdir(parents=True, exist_ok=False)
        write_json(attempt / "provenance.json", {"run_identity_sha256": self.identity_hash,
                   "case_id": case_id, "condition": condition, "attempt": len(attempts) + 1,
                   "allocated_at": datetime.now(timezone.utc).isoformat()})
        return attempt

    def started(self, attempt):
        write_json(attempt / "execution_timing.json", {
            "started_at": datetime.now(timezone.utc).isoformat()})

    def finish(self, attempt, record, trajectory=()):
        """Timeout and worker completion race: first terminal result wins."""
        with self.lock:
            timing_path = attempt / "execution_timing.json"
            timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else {}
            timing_key = "late_completed_at" if (attempt / "prediction.json").exists() else "terminal_recorded_at"
            timing[timing_key] = datetime.now(timezone.utc).isoformat()
            write_json(timing_path, timing)
            if (attempt / "prediction.json").exists():
                # A late thread remains attributable without replacing timeout.
                write_json(attempt / "late_completion.json", record)
                return False
            case_dir = attempt.parent.parent
            write_json(attempt / "prediction.json", record)
            text = "".join(json.dumps({"node": x}, ensure_ascii=False) + "\n" for x in trajectory)
            (attempt / "trajectory.jsonl").write_text(text, encoding="utf-8")
            write_json(case_dir / "prediction.json", record)
            (case_dir / "trajectory.jsonl").write_text(text, encoding="utf-8")
            return True

    def record(self, condition, case_id):
        p = self.case_dir(condition, case_id) / "prediction.json"
        if not p.exists():
            return failure_record(case_id, "missing_output: planned case has no result")
        return json.loads(p.read_text(encoding="utf-8"))

    def stats(self, condition, cases):
        first, total, wall, lower_bound = [], 0, 0.0, False
        for case in cases:
            paths = sorted((self.case_dir(condition, case.case_id) / "attempts").glob("*/prediction.json"))
            if not paths:
                first.append(False)
            for i, p in enumerate(paths):
                rec = json.loads(p.read_text(encoding="utf-8"))
                total += 1
                wall += rec.get("wall_seconds", 0.0)
                lower_bound |= rec.get("failure_detail", "").startswith(("case_timeout:", "interrupted_attempt:"))
                if i == 0:
                    first.append(rec.get("failure") is None and rec.get("final_behavior") is not None)
        return {"planned_cases": len(cases), "attempts": total,
                "first_attempt_valid_run_rate": sum(first) / len(cases),
                "attempt_wall_seconds_sum": wall, "wall_cost_is_lower_bound": lower_bound,
                "token_cost": None, "token_cost_note": "not instrumented; never interpreted as zero"}
