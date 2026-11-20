"""Per-case model-call timing/usage without prompt, response or credential logging."""
from datetime import datetime, timezone
import re
import threading
import time
from langchain_core.callbacks import BaseCallbackHandler


class CallAudit(BaseCallbackHandler):
    def __init__(self):
        self.calls = {}
        self.structured_failures = []
        self.lock = threading.Lock()

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        text = str(messages[0][0].content) if messages and messages[0] else ''
        role = re.search(r'Role:\s*([^\n]+)', text)
        with self.lock:
            self.calls[str(run_id)] = {'role': role.group(1)[:100] if role else 'team',
                'started_at': datetime.now(timezone.utc).isoformat(), '_start': time.monotonic(),
                'status': 'running', 'tokens': None, 'observed_retries': 0}

    def on_retry(self, retry_state, *, run_id, **kwargs):
        with self.lock:
            if str(run_id) in self.calls:
                self.calls[str(run_id)]['observed_retries'] += 1

    def _finish(self, run_id, status, tokens=None, error=None):
        with self.lock:
            call = self.calls.get(str(run_id))
            if call is None: return
            call.update(status=status, tokens=tokens, error_type=error,
                finished_at=datetime.now(timezone.utc).isoformat(),
                wall_seconds=round(time.monotonic()-call['_start'],6))

    def on_llm_end(self, response, *, run_id, **kwargs):
        usage = (response.llm_output or {}).get('token_usage')
        if not usage:
            for group in response.generations:
                for generation in group:
                    usage = getattr(getattr(generation,'message',None),'usage_metadata',None)
                    if usage: break
                if usage: break
        self._finish(run_id, 'completed', usage)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._finish(run_id, 'error', error=type(error).__name__)

    def structured_failure(self, role, attempt, error):
        with self.lock:
            self.structured_failures.append(dict(role=role, attempt=attempt,
                error_type=type(error).__name__, recorded_at=datetime.now(timezone.utc).isoformat()))

    def records(self):
        return [{k:v for k,v in call.items() if not k.startswith('_')} for call in self.calls.values()]


def attach_call_audit(llm, hooks):
    audit = CallAudit()
    instrumented = (getattr(llm, '_llm_type', '') != 'scripted-mock'
                    and hasattr(llm, 'model_copy') and
                    (getattr(llm, 'callbacks', None) is None or isinstance(llm.callbacks, list)))
    if instrumented:
        original = llm
        llm = llm.model_copy(update={'callbacks': list(llm.callbacks or [])+[audit]})
        for hook in (hooks.h1, hooks.h2):
            if getattr(hook, 'llm', None) is original:
                hook.llm = llm
    return llm, audit, instrumented


def record_structured_failure(llm, role, attempt, error):
    callbacks = getattr(llm, 'callbacks', None)
    for callback in callbacks if isinstance(callbacks, list) else []:
        if isinstance(callback, CallAudit):
            callback.structured_failure(role, attempt, error)
