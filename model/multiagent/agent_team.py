"""TalkHier hierarchical team skeleton (mechanism port from talkhier-main).

Preserved mechanisms:
  * recursive ``buildTeam()`` — nested teams are real ``AgentTeam`` instances
    with their own supervisor, routing, and independent history;
  * per-agent independent memory (no shared-memory mode at all);
  * the message / background / intermediate_output (M/B/I) channels are the
    real Supervisor <-> Member communication payload;
  * supervisors route via structured output (thoughts/messages/next/
    background/intermediate_output) with a bounded retry loop;
  * members are ReAct agents (see react_agent.py).

Payment adaptations (generic hooks, no payment logic lives here):
  * ``recorder``: optional callable invoked with the canonical node name on
    every actual agent execution (used for HF1/ASR trajectory instrumentation);
  * ``background_augmenter``: optional callable ``str -> str`` applied to the
    background text right before a prompt is built (used to propagate
    released dynamic environment events through the B channel).
"""
from __future__ import annotations

import ast
import functools
import json
from typing import Any, Callable, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator
from typing_extensions import Literal, TypedDict

from multiagent.llm import StructuredResponseError, classify_llm_error
from multiagent.react_agent import create_react_agent

SUPERVISOR_ATTEMPTS = 10
HISTORY_WINDOW = 15

Recorder = Optional[Callable[[str], None]]
Augmenter = Optional[Callable[[str], str]]


def _parse_intermediate_output(value: Any, fallback: Any) -> Any:
    """Parse a stringified intermediate output; fall back on failure."""
    if not isinstance(value, str):
        return value
    parsed = value
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        try:
            parsed = json.loads(value)
        except Exception:
            return fallback
    if parsed == {}:
        return fallback
    return parsed


class AgentTeam:
    def __init__(
        self,
        llm,
        member_list: Optional[list] = None,
        member_names: Optional[List[str]] = None,
        member_info: Optional[List[str]] = None,
        team_name: str = "Default",
        supervisor_name: str = "FINISH",
        intermediate_output_desc: str = "",
        is_main: bool = False,
        recorder: Recorder = None,
        background_augmenter: Augmenter = None,
        on_node_output: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
        verbose: bool = False,
        final_validator=None,
    ):
        self.llm = llm
        self.graph = None
        self.member_list = member_list or []
        self.member_info = member_info or []
        self.member_names = member_names or []
        self.team_name = team_name
        self.sup_name = supervisor_name
        self.intermediate_output_desc = intermediate_output_desc
        self.is_main = is_main
        self.recorder = recorder
        self.background_augmenter = background_augmenter
        self.on_node_output = on_node_output
        self.verbose = verbose
        self.final_validator = final_validator

        self.supervisor_node_name = f"{self.team_name} Supervisor"
        self.message_prompt = (
            f"If the next agent is one of {', '.join(self.member_names)}, give "
            f"detailed instructions and requests. If {supervisor_name}, report a "
            f"summary of all results."
        )
        self.thought_prompt = (
            "Output a detailed analysis on the most recent message. In detail, "
            "state what you think should be done next, and who you should contact "
            "next."
        )

    # ------------------------------------------------------------------ nodes
    def loadSupervisor(self, additional_prompt: str = ""):
        options = [self.sup_name] + self.member_names
        job_list = "\n".join(
            f"{mem}: {info}" for mem, info in zip(self.member_names, self.member_info)
        )
        # additional_prompt is literal text: escape braces so JSON examples in
        # prompts are not mistaken for template variables.
        additional_prompt = additional_prompt.replace("{", "{{").replace("}", "}}")
        prompt_list = [
            (
                "system",
                "You are a supervisor tasked with managing a conversation between "
                "the following workers: {members}. Given the following messages, "
                "respond with the worker to act next. Each worker will perform a "
                "task and respond with their results and status."
                "\nJoblist: \n{joblist}",
            ),
            ("system", "The current background is: {background}"),
            ("system", "Conversation History:"),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, output the following in this exact "
                "order:\n"
                "1. 'thoughts': {thought_prompt}\n"
                "2. 'messages': {message_prompt}\n"
                "3. Who should act next? Select one of: {options} and output as "
                "'next'. When you have determined that the final output is gained, "
                "report back with {finish} as 'next'.\n"
                "4. The detailed background of the problem you are trying to solve "
                "(given in the first message) as 'background'.\n"
                "5. The intermediate outputs to give as 'intermediate_output'.\n"
                + additional_prompt
                + "{error}"
            ),
        ]

        prompt = ChatPromptTemplate.from_messages(prompt_list).partial(
            options=str(options),
            members=", ".join(self.member_names),
            joblist=job_list,
            finish=self.sup_name,
            thought_prompt=self.thought_prompt,
            message_prompt=self.message_prompt,
        )
        is_main = self.is_main
        sup_node = self.supervisor_node_name
        sup_name = self.sup_name
        recorder = self.recorder
        augmenter = self.background_augmenter
        io_desc = self.intermediate_output_desc

        class RouteResponse(BaseModel):
            thoughts: str = Field(description=self.thought_prompt)
            next: Literal[*options]  # type: ignore[valid-type]
            messages: str = Field(description=self.message_prompt)
            intermediate_output: str = Field(description=io_desc)
            background: str

            @field_validator("messages", "intermediate_output", mode="before")
            def cast_to_string(cls, v):
                return str(v)

        def supervisor_agent(state: Dict[str, Any]) -> Dict[str, Any]:
            if recorder:
                recorder(sup_node)
            prev_history = state["history"]

            if is_main:
                background = ""
            else:
                bg = state["background"]
                background = bg.content if isinstance(bg, BaseMessage) else str(bg)
            if augmenter:
                background = augmenter(background)

            chain_input = {
                "messages": list(prev_history.get(sup_node, []))[-HISTORY_WINDOW:],
                "background": background,
                "error": "",
            }

            chain = prompt | self.llm.with_structured_output(RouteResponse)
            result = None
            validation_failures = 0
            last_exc: Optional[BaseException] = None
            for _ in range(SUPERVISOR_ATTEMPTS):
                try:
                    result = chain.invoke(chain_input)
                    if is_main and result.next == sup_name and self.final_validator:
                        candidate = _parse_intermediate_output(result.intermediate_output, state["intermediate_output"])
                        if candidate == "":
                            candidate = state["intermediate_output"]
                        problem = self.final_validator(candidate)
                        if problem:
                            validation_failures += 1
                            if validation_failures >= 3:
                                raise StructuredResponseError("PARSE_FAILURE", "final validation: " + problem)
                            chain_input["error"] = "\nFinal output needs correction: " + problem
                            if augmenter:
                                chain_input["background"] = augmenter("")
                            result = None
                            continue
                    break
                except Exception as exc:  # noqa: BLE001 - retried, then classified
                    if isinstance(exc, StructuredResponseError):
                        raise
                    from multiagent.call_audit import record_structured_failure
                    record_structured_failure(self.llm, sup_node, _ + 1, exc)
                    last_exc = exc
                    print(f"[{sup_node}] routing error, retrying: {exc}")
                    chain_input["error"] = (
                        "\n\nDouble check that 'next' is one of: " + str(options)
                    )
            if result is None:
                raise StructuredResponseError(
                    classify_llm_error(last_exc), str(last_exc)
                )

            new_msg = AIMessage(content=result.messages, name=sup_node.replace(" ", "_"))
            if not (
                "Intermediate Output" in new_msg.content
                or "Final Output" in new_msg.content
            ):
                if result.intermediate_output in ["", "{{}}"]:
                    result.intermediate_output = state["intermediate_output"]
                if str(result.intermediate_output) not in new_msg.content:
                    new_msg.content = (
                        new_msg.content
                        + "\n\nFinal Output: "
                        + str(result.intermediate_output)
                    )

            intermediate = result.intermediate_output
            if isinstance(intermediate, str):
                intermediate = _parse_intermediate_output(
                    intermediate, state["intermediate_output"]
                )

            own_msg = new_msg.model_copy()
            own_msg.content = "{Thoughts: " + result.thoughts + "}\n\n" + own_msg.content
            new_history = {
                **prev_history,
                sup_node: prev_history.get(sup_node, []) + [own_msg],
                result.next: prev_history.get(result.next, []) + [new_msg.model_copy()],
            }

            return {
                "intermediate_output": intermediate,
                "messages": new_msg,
                "background": AIMessage(content=result.background, name=sup_node),
                "next": result.next,
                "history": new_history,
            }

        return supervisor_agent

    # ------------------------------------------------------------------ graph
    def createStateGraph(self, additional_prompt: str = ""):
        class TeamState(TypedDict):
            history: Dict[str, List[BaseMessage]]
            messages: BaseMessage
            background: BaseMessage
            intermediate_output: Any
            next: str

        workflow = StateGraph(TeamState)
        for member, member_name in zip(self.member_list, self.member_names):
            workflow.add_node(member_name, member)
        workflow.add_node(
            self.supervisor_node_name, self.loadSupervisor(additional_prompt)
        )
        for member in self.member_names:
            workflow.add_edge(member, self.supervisor_node_name)

        conditional_map = {k: k for k in self.member_names}
        conditional_map[self.sup_name] = END
        workflow.add_conditional_edges(
            self.supervisor_node_name, lambda x: x["next"], conditional_map
        )
        workflow.add_edge(START, self.supervisor_node_name)

        self.graph = workflow.compile(debug=False)
        return functools.partial(
            AgentTeam.run_team,
            graph=self.graph,
            team_name=self.team_name,
            on_node_output=self.on_node_output,
            verbose=self.verbose,
        )

    # ------------------------------------------------------------------ driver
    # Channel names of TeamState; used to detect "values"-shaped chunks that
    # langgraph emits (instead of {node: update}) when a team graph is
    # streamed inside a parent graph's node (nested team execution).
    TEAM_CHANNELS = frozenset(
        {"history", "messages", "background", "intermediate_output", "next"}
    )

    @staticmethod
    def run_team(state, config, graph, team_name, on_node_output=None, verbose=False):
        """Stream the team graph to completion; return the terminal state slice.

        Chunk shape handling:
          * top-level stream: ``{node_name: node_update}`` — node routing and
            ``on_node_output`` fire per node;
          * nested stream (inside a parent node): full-state chunks — the last
            chunk is the nested team's final state, which doubles as the
            parent node's update.

        Either way the final ``result`` carries ``intermediate_output`` /
        ``messages`` / ``history`` / ``next`` produced by the supervisor that
        closed the run (its ``next`` is the team's ``sup_name``/FINISH).
        """
        result = None
        for chunk in graph.stream(state, config):
            if "__end__" in chunk:
                continue
            if set(chunk.keys()) <= AgentTeam.TEAM_CHANNELS:
                result = chunk
                continue
            node, update = next(iter(chunk.items()))
            result = update
            if on_node_output:
                on_node_output(team_name, node, update)
            if verbose:
                nxt = update.get("next", "?") if isinstance(update, dict) else "?"
                print(f"[team={team_name}] node={node} -> next={nxt}")
        return result


class ReactAgent:
    """Factory for ReAct member nodes compatible with AgentTeam."""

    def __init__(
        self,
        llm,
        intermediate_output_desc: str = "",
        tool_resolver: Optional[Callable[[list], list]] = None,
        recorder: Recorder = None,
        background_augmenter: Augmenter = None,
        verbose: bool = False,
        final_validator=None,
    ):
        self.llm = llm
        self.intermediate_output_desc = intermediate_output_desc
        self.tool_resolver = tool_resolver or (lambda selector: list(selector))
        self.recorder = recorder
        self.background_augmenter = background_augmenter
        self.verbose = verbose

    def loadMember(self, name: str, member_tools: list, member_prompt: str, sup_name: str):
        resolved_tools = self.tool_resolver(member_tools)
        # member_prompt is literal text; escape braces so JSON examples are not
        # mistaken for template variables.
        member_prompt = member_prompt.replace("{", "{{").replace("}", "}}")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", member_prompt),
                ("system", "Background: {background}"),
                ("system", "Conversation History:"),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        io_desc = self.intermediate_output_desc

        class MemberResponse(BaseModel):
            intermediate_output: str = Field(description=io_desc)

        agent = create_react_agent(
            self.llm,
            tools=resolved_tools,
            response_schema=MemberResponse,
            response_format=lambda result: {
                "intermediate_output": result.intermediate_output
            },
            state_modifier=prompt,
        )
        return functools.partial(
            ReactAgent.agent_node,
            agent=agent,
            name=name,
            sup_name=sup_name,
            recorder=self.recorder,
            augmenter=self.background_augmenter,
            verbose=self.verbose,
        )

    @staticmethod
    def agent_node(state, agent, name, sup_name, recorder=None, augmenter=None, verbose=False):
        if recorder:
            recorder(name)
        prev_history = state["history"]
        own_history = list(prev_history.get(name, []))

        bg = state.get("background")
        background = bg.content if isinstance(bg, BaseMessage) else (bg or "")
        if augmenter:
            background = augmenter(background)

        member_state = {
            "messages": own_history[-HISTORY_WINDOW:],
            "background": background,
        }

        result = None
        for chunk in agent.stream(member_state, {}, stream_mode="values"):
            result = chunk
            if verbose:
                msgs = chunk.get("messages", [])
                if msgs:
                    print(f"[{name}] messages={len(msgs)}")
        if result is None:
            raise StructuredResponseError("PROVIDER_FAILURE", "member produced no output")

        final_messages = result["messages"]
        new_msg = AIMessage(
            content=final_messages[-1].content, name=name.replace(" ", "_")
        )
        intermediate = result.get("intermediate_output")
        if intermediate is not None and "Final Output" not in new_msg.content:
            if str(intermediate) not in new_msg.content:
                new_msg.content = new_msg.content + "\n\nFinal Output: " + str(intermediate)

        intermediate = _parse_intermediate_output(
            intermediate, state["intermediate_output"]
        )

        new_history = {
            **prev_history,
            sup_name: prev_history.get(sup_name, []) + [new_msg.model_copy()],
            name: prev_history.get(name, []) + [new_msg.model_copy()],
        }

        return {
            "intermediate_output": intermediate,
            "messages": new_msg,
            "background": AIMessage(content=background, name=name),
            "next": state["next"],
            "history": new_history,
        }


def buildTeam(team_information, react_factory, intermediate_output_desc, recorder=None,
              background_augmenter=None, on_node_output=None, verbose=False,
              final_validator=None):
    """Recursively build the (possibly nested) team graph."""
    team_list: List[Any] = []
    member_info: List[str] = []
    member_names: List[str] = []

    for key, spec in team_information.items():
        if not isinstance(spec, dict):
            continue
        if "team" in spec:
            team_list.append(
                buildTeam(
                    spec,
                    react_factory,
                    intermediate_output_desc,
                    recorder=recorder,
                    background_augmenter=background_augmenter,
                    on_node_output=on_node_output,
                    verbose=verbose,
                )
            )
            member_names.append(spec["team"] + " Supervisor")
            member_info.append(spec["prompt"])
        else:
            team_list.append(
                react_factory.loadMember(
                    key,
                    spec.get("tools", []),
                    spec["prompt"],
                    team_information["team"] + " Supervisor",
                )
            )
            member_names.append(key)
            member_info.append(spec["prompt"])

    return AgentTeam(
        llm=react_factory.llm,
        member_list=team_list,
        member_info=member_info,
        member_names=member_names,
        team_name=team_information["team"],
        supervisor_name=team_information["return"],
        intermediate_output_desc=intermediate_output_desc,
        is_main=bool(team_information.get("is_main", False)),
        final_validator=final_validator,
        recorder=recorder,
        background_augmenter=background_augmenter,
        on_node_output=on_node_output,
        verbose=verbose,
    ).createStateGraph(additional_prompt=team_information.get("additional_prompt", ""))
