"""Shared spot-price / fee economics for the payment environment (M6).

One implementation, two consumers — the simulator's receipt builder
(``src/payment/environment.py``) and the dataset validator's P6
satisfiability check (``scripts/validate_dataset_v4.py``) — so the estimate
the team sees on a receipt can never drift from the estimate the dataset
was validated against.

Price model (deterministic, no randomness):
  * direct pair ``<asset>/<target>`` in spot_prices wins;
  * otherwise the USD legs ``<asset>/USD`` and ``<target>/USD`` are crossed;
  * otherwise the estimate is unavailable (None).

Fee model (simulated venue, domain defaults — see DEFAULT_* constants):
  * ``gas_fee`` = gas_price_gwei x gas_used, quoted in gwei, converted into
    the transacted asset via the USD legs (ETH/USD falls back to
    DEFAULT_ETH_USD when the market layer does not list it);
  * ``protocol_fee`` = amount x protocol_fee_bps (default 0 — the venue
    currently charges no proportional fee);
  * ``total_fee`` = gas_fee + protocol_fee, in the transacted asset unit.

The defaults are deliberately tiny (a cheap L2-style venue) so the P6
fee-cap commitments harvested from upstream conversations stay satisfiable;
they are documented in environment_query's field_semantics.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# Domain defaults for the simulated venue. gas_used=100 reflects a cheap
# L2-style execution environment (documented in field_semantics); at the
# template gas price of 5 gwei this yields a ~$0.0015 fee, which keeps the
# dataset's tightest fee caps (>= 0.001616 USDC) satisfiable.
DEFAULT_GAS_USED = 100
DEFAULT_PROTOCOL_FEE_BPS = 0.0
DEFAULT_ETH_USD = 3000.0
GWEI_TO_NATIVE = 1e-9

# Shopping (ecommerce-payment) economics — deterministic, catalog-driven.
# Shipping fee tiers by unit price (USD), applied per order when the SKU is
# not ships_free; coupon_usd is a flat per-order discount from the catalog.
SHIPPING_TIERS = ((50.0, 4.99), (150.0, 8.99), (float("inf"), 13.99))


def spot_rate(asset: str, target: str,
              prices: Dict[str, Any]) -> Optional[float]:
    """Units of ``target`` per unit of ``asset`` at spot, or None."""
    direct = prices.get(f"{asset}/{target}")
    if isinstance(direct, (int, float)):
        return float(direct)
    src_usd = prices.get(f"{asset}/USD")
    tgt_usd = prices.get(f"{target}/USD")
    if (isinstance(src_usd, (int, float)) and isinstance(tgt_usd, (int, float))
            and tgt_usd != 0):
        return float(src_usd) / float(tgt_usd)
    return None


def estimated_output(operation: str, amount: Any, asset: Optional[str],
                     target: Optional[str],
                     prices: Dict[str, Any]) -> Optional[float]:
    """Estimated proceeds of the transaction in the incoming asset unit.

    swap: amount x spot rate (asset -> target). Non-swap operations
    (deposit/withdraw/borrow/repay) move the same asset in and out, so the
    estimated output equals the amount. Returns None when inputs are
    missing or no price is available.
    """
    if not isinstance(amount, (int, float)):
        return None
    if operation == "swap":
        if not (asset and target):
            return None
        rate = spot_rate(asset, target, prices)
        return amount * rate if rate is not None else None
    return float(amount)


def asset_per_usd(asset: str, prices: Dict[str, Any]) -> Optional[float]:
    """Units of ``asset`` per 1 USD (1 / <asset>/USD), or None."""
    usd = prices.get(f"{asset}/USD")
    if isinstance(usd, (int, float)) and usd != 0:
        return 1.0 / float(usd)
    return None


def gas_fee_in_asset(gas_price_gwei: Any, prices: Dict[str, Any],
                     asset: Optional[str],
                     gas_used: int = DEFAULT_GAS_USED) -> Optional[float]:
    """Gas cost converted into ``asset`` units.

    gas_fee_eth = gas_price_gwei x gas_used x 1e-9; the ETH value is crossed
    into USD (ETH/USD, falling back to DEFAULT_ETH_USD) and then into the
    target asset via <asset>/USD. Returns None when the asset has no USD
    leg (or no asset is given).
    """
    if not (isinstance(gas_price_gwei, (int, float)) and asset):
        return None
    fee_eth = float(gas_price_gwei) * float(gas_used) * GWEI_TO_NATIVE
    eth_usd = prices.get("ETH/USD")
    if not isinstance(eth_usd, (int, float)):
        eth_usd = DEFAULT_ETH_USD
    fee_usd = fee_eth * float(eth_usd)
    if asset == "ETH":
        return fee_eth
    per_usd = asset_per_usd(asset, prices)
    return fee_usd * per_usd if per_usd is not None else None


def fee_breakdown(operation: str, parameters: Dict[str, Any],
                  facts: Dict[str, Any],
                  gas_used: int = DEFAULT_GAS_USED,
                  protocol_fee_bps: float = DEFAULT_PROTOCOL_FEE_BPS
                  ) -> Dict[str, Any]:
    """Compute the receipt's economic fields for a successful simulation.

    Returns a dict with ``estimated_output``, ``gas_fee``, ``protocol_fee``
    and ``total_fee`` (any of which may be None when not computable), plus
    the ``fee_unit`` (the transacted asset) and the ``gas_used`` /
    ``protocol_fee_bps`` assumptions for auditability.
    """
    prices = facts.get("spot_prices", {}) or {}
    amount = parameters.get("amount")
    asset = parameters.get("asset")
    target = parameters.get("target_asset")

    est = estimated_output(operation, amount, asset, target, prices)
    gas_fee = gas_fee_in_asset(facts.get("gas_price_gwei"), prices, asset,
                               gas_used)
    protocol_fee = (float(amount) * protocol_fee_bps / 10_000.0
                    if isinstance(amount, (int, float)) else None)
    total = None
    if gas_fee is not None or protocol_fee is not None:
        total = (gas_fee or 0.0) + (protocol_fee or 0.0)
    return {
        "estimated_output": est,
        "gas_fee": gas_fee,
        "protocol_fee": protocol_fee,
        "total_fee": total,
        "fee_unit": asset,
        "gas_used": gas_used,
        "protocol_fee_bps": protocol_fee_bps,
    }


def shipping_fee_for(unit_price: Any, ships_free: Any) -> Optional[float]:
    """Flat-tier shipping fee for one order (ecommerce-payment domain).

    Free when the SKU carries ``ships_free``; otherwise a deterministic tier
    by unit price (SHIPPING_TIERS). Returns None when the price is missing.
    """
    if ships_free:
        return 0.0
    if not isinstance(unit_price, (int, float)):
        return None
    for ceiling, fee in SHIPPING_TIERS:
        if float(unit_price) < ceiling:
            return fee
    return SHIPPING_TIERS[-1][1]


def purchase_breakdown(product: Dict[str, Any], quantity: Any
                       ) -> Dict[str, Any]:
    """Receipt economics for a ``shop.purchase`` simulation.

    total_price = unit_price x quantity + shipping_fee - coupon_discount,
    all computed deterministically from the product-catalog record
    (``price_usd`` / ``ships_free`` / ``coupon_usd``). Shared by the
    simulator's receipt builder and the v5 dataset validator (one
    implementation, two consumers — the M6 principle). Any component that
    cannot be computed yields None fields instead of guessing.
    """
    unit_price = product.get("price_usd")
    qty = quantity if isinstance(quantity, (int, float)) else None
    shipping = shipping_fee_for(unit_price, product.get("ships_free"))
    coupon = product.get("coupon_usd")
    coupon_discount = float(coupon) if isinstance(coupon, (int, float)) else 0.0
    total = None
    if isinstance(unit_price, (int, float)) and qty is not None and shipping is not None:
        total = round(float(unit_price) * float(qty) + shipping - coupon_discount, 2)
    return {
        "unit_price": float(unit_price) if isinstance(unit_price, (int, float)) else None,
        "quantity": qty,
        "shipping_fee": shipping,
        "coupon_discount": coupon_discount,
        "total_price": total,
        "currency": "USD",
    }
