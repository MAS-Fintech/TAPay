"""Minimal ReAct member-agent loop (TalkHier mechanism, cleaned rewrite).

Semantics preserved from the upstream talkhier-main react_agent.py:
  * ``agent`` node calls the tool-bound chat model;
  * if the last AI message carries tool calls, route to ``tools`` (ToolNode)
    and loop back to ``agent``;
  * otherwise route to ``respond``, which produces the structured
    ``intermediate_output`` via ``model.with_structured_output`` with a
    bounded retry loop, then ends.

Differences from upstream: ToolNode-only construction (no deprecated
ToolExecutor), explicit StructuredResponseError failure classification, and
no dead code paths.
"""
from __future__ import annotations

from typing import Callable, Literal, Optional, Sequence, Type, Union

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep
from langgraph.prebuilt.tool_node import ToolNode
from pydantic import BaseModel
from typing_extensions import Annotated, TypedDict

from multiagent.llm import StructuredResponseError, classify_llm_error

STRUCTURED_OUTPUT_ATTEMPTS = 10


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    background: BaseMessage
    intermediate_output: str
    is_last_step: IsLastStep


StateModifier = Union[
    str,
    Callable[["AgentState"], Sequence[BaseMessage]],
    Runnable["AgentState", Sequence[BaseMessage]],
]


def _as_state_modifier_runnable(state_modifier):
    if state_modifier is None:
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(lambda state: state["messages"])
    if isinstance(state_modifier, str):
        from langchain_core.messages import SystemMessage

        sys_msg = SystemMessage(content=state_modifier)
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(lambda state: [sys_msg] + list(state["messages"]))
    return state_modifier


def create_react_agent(
    model: LanguageModelLike,
    tools: Sequence[BaseTool],
    response_schema: Optional[Type[BaseModel]],
    response_format: Optional[Callable] = None,
    *,
    state_modifier: Optional[StateModifier] = None,
    debug: bool = False,
):
    tool_node = ToolNode(tools)
    tool_classes = list(tool_node.tools_by_name.values())
    model = model.bind_tools(tool_classes)

    preprocessor = _as_state_modifier_runnable(state_modifier)
    model_runnable = preprocessor | model

    if response_schema is not None:
        structured_runnable = preprocessor | model.with_structured_output(
            response_schema
        )

        def final_response(state: AgentState, config: RunnableConfig):
            last_exc: Optional[BaseException] = None
            for _ in range(STRUCTURED_OUTPUT_ATTEMPTS):
                try:
                    result = structured_runnable.invoke(state, config)
                    if response_format is not None:
                        return response_format(result)
                    return {"intermediate_output": result.intermediate_output}
                except Exception as exc:  # noqa: BLE001 - retried, then classified
                    last_exc = exc
                    print(f"[respond] error, retrying: {exc}")
            raise StructuredResponseError(
                classify_llm_error(last_exc), str(last_exc)
            )

        def should_continue(state: AgentState) -> Literal["tools", "respond"]:
            last_message = state["messages"][-1]
            if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
                return "respond"
            return "tools"
    else:

        def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
            last_message = state["messages"][-1]
            if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
                return "__end__"
            return "tools"

    def call_model(state: AgentState, config: RunnableConfig):
        response = model_runnable.invoke(state, config)
        if (
            state.get("is_last_step")
            and isinstance(response, AIMessage)
            and response.tool_calls
        ):
            response = AIMessage(
                id=response.id,
                content="Sorry, need more steps to process this request.",
            )
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    if response_schema is not None:
        workflow.add_node("respond", final_response)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)

    should_return_direct = {t.name for t in tool_classes if t.return_direct}

    def route_tool_responses(state: AgentState) -> Literal["agent", "__end__"]:
        from langchain_core.messages import ToolMessage

        for m in reversed(state["messages"]):
            if not isinstance(m, ToolMessage):
                break
            if m.name in should_return_direct:
                return "__end__"
        return "agent"

    if should_return_direct:
        workflow.add_conditional_edges("tools", route_tool_responses)
    else:
        workflow.add_edge("tools", "agent")
        if response_schema is not None:
            workflow.add_edge("respond", "__end__")

    return workflow.compile(debug=debug)
