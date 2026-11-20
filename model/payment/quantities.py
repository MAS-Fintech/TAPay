"""Unit-safe receipt arithmetic; prices must belong to the receipt's state."""
import math
from payment.economics import DEFAULT_ETH_USD, spot_rate


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def receipt_quantity(receipt, field, unit, prices=None):
    value = receipt.get(field)
    params = receipt.get('parameters') or {}
    source = (receipt.get('fee_unit') if field in ('total_fee', 'gas_fee') else
              receipt.get('currency', 'USD') if field == 'total_price' else
              params.get('target_asset') if receipt.get('operation') == 'swap' else params.get('asset'))
    evidence = {'source_value': value, 'source_unit': source, 'unit': unit}
    if not finite(value) or not source or not unit:
        return None, dict(evidence, reason='missing/nonfinite value or unit')
    if source == unit:
        return float(value), evidence
    # A fee may be expressed in a different asset. Product totals/output asset
    # requirements do not silently become FX transactions.
    if field not in ('total_fee', 'gas_fee') or prices is None:
        return None, dict(evidence, reason='incomparable units without conversion evidence')
    prices = dict(prices)
    prices.setdefault('ETH/USD', DEFAULT_ETH_USD)  # documented simulator tariff
    prices.setdefault('USD/USD', 1.0)
    rate = spot_rate(source, unit, prices)
    if not finite(rate) or rate <= 0:
        return None, dict(evidence, reason='missing/invalid same-state conversion rate')
    converted = value * rate
    if not finite(converted):
        return None, dict(evidence, reason='nonfinite conversion')
    return converted, dict(evidence, rate=rate, converted=converted)
