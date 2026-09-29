"""Dynamic Decision Boundary Agent module.

Design inspirations are adaptations, not reproductions:
- ICC-Bandit: contextual constraints inform action selection. This implementation
  uses deterministic monotone boundary rules, not a trained bandit.
- Slow/fast loops: final-candidate control and event-driven rechecks.
- Field-level uncertainty and version-bound execution evidence.

The policy is deterministic and auditable. Automatic execution requires
sufficient intent, successful validation, feasible environment facts and
current evidence. Missing current receipts may trigger bounded simulation.
Substantive failures or unresolved evidence prevent execution. The policy
never promotes a prior BLOCK or REQUEST_CLARIFICATION to EXECUTE.
"""
