"""Reproducible case blocks and a host-wide lease for this project's provider jobs."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import random
import tempfile


def balanced_tasks(tasks, seed):
    grouped = {}
    for task in tasks:
        grouped.setdefault(task[1].case_id, []).append(task)
    rng = random.Random(seed)
    keys = sorted(grouped)
    rng.shuffle(keys)
    result = []
    for key in keys:
        block = sorted(grouped[key], key=lambda t: t[0])
        rng.shuffle(block)
        result.extend(block)
    return result


@contextmanager
def provider_lease(config, *, mock=False, lock_root=None, lease_slot=0):
    """Reject overlapping local CLIs sharing an endpoint; never log credentials.

    OS locks are released on process exit, including crashes. This coordinates
    these runners on this host, not arbitrary clients or remote machines.

    ``lease_slot`` (operational only, not recorded in run identity): slot 0 is
    the default global lease. Slots > 0 take a per-slot lease, allowing N
    intentional parallel matrices on the same endpoint when the operator has
    provisioned per-account capacity accordingly (e.g. two accounts via
    api_keys_extra, each matrix capped at a safe worker count).
    """
    if mock:
        yield
        return
    sec = config['LLM']
    endpoint = (sec.get('base_url') or sec.get('endpoint') or 'default').rstrip('/').lower()
    key_src = sec.get('provider', '') + ':' + endpoint
    if lease_slot:
        key_src += f"|slot{int(lease_slot)}"
    key = hashlib.sha256(key_src.encode()).hexdigest()
    root = Path(lock_root) if lock_root else Path(tempfile.gettempdir())/'tapay-provider-leases'
    root.mkdir(parents=True, exist_ok=True)
    handle = (root/(key+'.lock')).open('a+b')
    try:
        handle.seek(0,2)
        if not handle.tell():
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another local experiment is using this provider endpoint; '
                               'wait for it to finish, then run the next domain.') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        handle.close()
