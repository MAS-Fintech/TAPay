"""Canonical spelling of known payment methods; never selects an alternative.

This is lexical normalization of the user's field, not authorization. The live
service policy and simulator still decide whether the canonical method is allowed.
SKU, action id, receipt parameters, and unknown method names remain exact.
"""

_ALIASES = {
    'paypal': 'paypal',
    'credit card': 'credit_card', 'credit_card': 'credit_card',
    'debit card': 'debit_card', 'debit_card': 'debit_card',
}


def canonical_payment_method(value):
    if not isinstance(value, str):
        return value
    return _ALIASES.get(' '.join(value.split()).casefold(), value)
