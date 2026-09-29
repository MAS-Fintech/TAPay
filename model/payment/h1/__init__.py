"""Structured Intent Agent module.

Design inspirations are adaptations, not reproductions:
- Tagliaferro: staged extraction, verifiability classification and explicit
  syntax/intent checks with traceable intermediate artifacts.
- Rastogi SGD: classify the scheme/domain before reading required fields.
- Wang AwN: map IMKI/IMR/IwE/IBTC issues to missing, ambiguous, invalid and
  unsupported field states; execution-relevant uncertainty triggers clarification.
- Suri: represent uncertainty per field instead of as a single confidence score.
"""
