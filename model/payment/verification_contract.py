"""Shared evidence contract for this offline simulator, independent of H toggles.

These are benchmark semantics, not guarantees about settlement on a real chain.
Keep hard/soft modality separate from the kind of evidence that verifies it.
"""

ECONOMIC_FIELDS = {
    "min_output": ("estimated_output", "min"),
    "total_fee": ("total_fee", "max"),
    "total_price": ("total_price", "max"),
}

VERIFICATION_CONTRACT = """
Offline simulation verification contract (authoritative for every role):
- Hardness and verification method are separate. An inviolable constraint need not be an action-catalog parameter. Keep its strength; use the evidence specified below. Never add non-catalog parameters to simulate_transaction.
- A SIMULATED_EXECUTED receipt matching the action, complete parameters, CURRENT state and current intent/authorization is sufficient outcome evidence here. For min_output, compare receipt.estimated_output >= the bound in the incoming asset; for total_fee, compare receipt.total_fee <= the cap in fee_unit; for shopping total_price, compare receipt.total_price <= the USD cap, including shipping and coupons. Boundary tolerance is math.isclose(rel_tol=1e-6, abs_tol=1e-9). These comparisons verify even inviolable economic commitments. Do not demand an extra enforcement parameter, an on-chain guarantee, a worst-case slippage deduction, or another fee deduction from estimated_output. A satisfied receipt bound is not an omitted constraint.
- Compare quantities only in the same unit. Fee conversion may use the matching state's spot prices and documented simulator tariff; missing conversion evidence is unverified, not a numerical violation or success. Never compare raw USDC fees to an ETH cap.
- Missing, uncomputable, mismatched-unit or stale evidence does not establish satisfaction. Obtain the applicable tool evidence; never interpret a not-applicable rule PASS as verified. A violated hard bound remains a failure. E1 checks the user's bound and delegates receipt arithmetic to E3; absence of a min_output/fee/budget parameter alone is not FAIL. E3 verifies the actual receipt values. Supervisors and the Revisor apply the same contract to evaluator and H2 observations.
- amount_limit is checked against the proposed amount with its stated min/max/exact direction, in the environment's documented amount units. slippage_cap uses the slippage parameter in percent; gas_cap uses current gas_price_gwei in gwei. No duplicate enforcement fields are required.
- deadline uses time_window_seconds/deadline_at at the proposal and final_decision top level, resolved against conversation_start_at/timezone; preserve them and check the clock for expiry. They are not simulator parameters. remaining_time_s is a runtime budget, not the user's deadline. A receipt does not prove a real-world settlement-time guarantee.
- Quote receipt_id exactly from the tool result; never abbreviate it. The final decision is a JSON object with final_behavior, rationale, and (for EXECUTE) action_id, operation, parameters, receipt_id. When state changes, revisit prior conclusions; do not repeat an old budget or quantity from superseded intent.
- A cached receipt is reusable if all bindings above remain current; cached does not mean stale. Compare versions and bindings, not an earlier event's wording. A real state/intent change requires fresh evidence; missing or stale evidence calls for tool refresh, not a claim of fundamental infeasibility.
- Explicit hard requirements outside these supported representations (for example a mandatory private/MEV route or a specific execution-price guarantee) remain unresolved; do not soften them merely because the catalog cannot express them. Optional route hints remain preferences. No individual check grants EXECUTE: all applicable intent, authorization, feasibility and receipt checks must pass.
"""
