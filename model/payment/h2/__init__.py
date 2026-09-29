"""Transaction-Intent Alignment Agent module.

Design inspirations are adaptations, not reproductions:
- Arbiter: deterministic checks precede independent semantic verification,
  with audit evidence and violation classifications.
- RTADev: intermediate proposals and evaluation artifacts must pass checkpoints
  before downstream use; deterministic gates enforce hard constraints.
- Version binding: receipts must match the current state; stale evidence fails.

This agent independently checks the conversation, proposal, receipts and
observed environment. It does not rebuild the structured intent maintained
by the Structured Intent Agent.
"""
