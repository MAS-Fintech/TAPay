"""Restricted payment tools exposed to member agents.

Two live tools wrap the runner-owned EnvironmentLedger; ``output_tool`` is a
write-only thinking channel kept from upstream TalkHier for evaluators.
Every tool call is logged into the ledger for audit; failures are returned as
deterministic error strings (never exceptions) so the ReAct loop continues.
"""
from __future__ import annotations

import json
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, field_validator

from payment.environment import EnvironmentLedger


class EnvironmentQueryInput(BaseModel):
    scope: str = Field(
        default="current",
        description="Which environment snapshot to read: 'current' (default) "
        "or 'initial'. The 'initial' snapshot is rejected as stale once an "
        "environment event has been released.",
    )


class EnvironmentQueryTool(BaseTool):
    name: str = "environment_query"
    description: str = (
        "Query the offline payment environment. Returns the network, state "
        "version, conversation clock, the action catalog (action ids, "
        "operations and parameter schemas), the asset catalog, and current "
        "facts such as balance, max_amount, quote_valid, plan_valid, "
        "route_available, approval_present, risk_score, remaining_budget and "
        "remaining_time_s. Input: optional scope ('current' or 'initial')."
    )
    args_schema: Type[BaseModel] = EnvironmentQueryInput
    ledger: EnvironmentLedger

    def _run(
        self, scope: str = "current", run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        return self.ledger.environment_query(scope)


class SimulateInput(BaseModel):
    action_id: str = Field(
        description="The exact action id from the current action catalog."
    )
    parameters: str = Field(
        description="JSON object string with the action parameters, e.g. "
        '{"amount": 100.0, "asset": "ETH", "target_asset": "USDT", '
        '"slippage": 0.5}. Required fields and bounds come from the action '
        "catalog."
    )

    @field_validator("parameters", mode="before")
    def cast_to_string(cls, v):
        return v if isinstance(v, str) else json.dumps(v)


class SimulateTransactionTool(BaseTool):
    name: str = "simulate_transaction"
    description: str = (
        "Simulate a payment transaction against the current environment state "
        "and return a receipt. Validates the parameters against the action "
        "catalog and checks approval, balance, quote/plan validity and route "
        "availability. Simulation calls are budgeted; an exact proposal may "
        "be simulated only once per environment state version."
    )
    args_schema: Type[BaseModel] = SimulateInput
    ledger: EnvironmentLedger

    def _run(
        self,
        action_id: str,
        parameters: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            params = json.loads(parameters) if isinstance(parameters, str) else parameters
        except Exception as exc:
            return f"ERROR: parameters is not valid JSON: {exc}"
        return self.ledger.simulate(action_id, params)


class OutputInput(BaseModel):
    in_str: str = Field(description="Thoughts to write down; nothing is returned.")

    @field_validator("in_str", mode="before")
    def cast_to_string(cls, v):
        return str(v)


class ShopSearchInput(BaseModel):
    query: str = Field(
        default="",
        description="Free-text query; matches SKU title/brand/model "
        "(case-insensitive substring).",
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional exact category filter.",
    )
    state_version: Optional[int] = Field(
        default=None,
        description="Optional pin to a previously returned state_version; a "
        "pinned request against a superseded state is rejected as stale.",
    )


class ShopSearchProductsTool(BaseTool):
    name: str = "shop_search_products"
    description: str = (
        "Search the product catalog (ecommerce-payment environments only). "
        "Deterministically returns the catalog slice matching the query "
        "(title/brand/model substring, optional category filter), sorted by "
        "price ascending, with the current state_version. Product "
        "information is available ONLY through the shop tools. After an "
        "environment event the earlier results are stale: re-run the search."
    )
    args_schema: Type[BaseModel] = ShopSearchInput
    ledger: EnvironmentLedger

    def _run(
        self,
        query: str = "",
        category: Optional[str] = None,
        state_version: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return self.ledger.shop_search_products(query, category, state_version)


class ShopDetailsInput(BaseModel):
    sku: str = Field(description="The exact SKU to inspect.")
    state_version: Optional[int] = Field(
        default=None,
        description="Optional pin to a previously returned state_version; a "
        "pinned request against a superseded state is rejected as stale.",
    )


class ShopGetProductDetailsTool(BaseTool):
    name: str = "shop_get_product_details"
    description: str = (
        "Return the full current catalog record for one SKU (price, stock, "
        "seller, seller risk, shipping and coupon fields) with the current "
        "state_version. Ecommerce-payment environments only."
    )
    args_schema: Type[BaseModel] = ShopDetailsInput
    ledger: EnvironmentLedger

    def _run(
        self,
        sku: str,
        state_version: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return self.ledger.shop_get_product_details(sku, state_version)


class OutputTool(BaseTool):
    name: str = "output_tool"
    description: str = (
        "A tool to simply write your thoughts. Nothing will be returned for output."
    )
    args_schema: Type[BaseModel] = OutputInput

    def _run(
        self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        return ""


# Selector ids used in prompts.py team definitions.
TOOL_ENVIRONMENT = "environment_query"
TOOL_SIMULATE = "simulate_transaction"
TOOL_OUTPUT = "output_tool"
TOOL_SHOP_SEARCH = "shop_search_products"
TOOL_SHOP_DETAILS = "shop_get_product_details"


def get_payment_tools(selector, ledger: EnvironmentLedger):
    tools = []
    if TOOL_ENVIRONMENT in selector:
        tools.append(EnvironmentQueryTool(ledger=ledger))
    if TOOL_SIMULATE in selector:
        tools.append(SimulateTransactionTool(ledger=ledger))
    if TOOL_SHOP_SEARCH in selector:
        tools.append(ShopSearchProductsTool(ledger=ledger))
    if TOOL_SHOP_DETAILS in selector:
        tools.append(ShopGetProductDetailsTool(ledger=ledger))
    if TOOL_OUTPUT in selector:
        tools.append(OutputTool())
    return tools
