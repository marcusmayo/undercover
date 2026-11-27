"""The claim agent as a LangGraph graph: it decides, waits for a sign-off and pays.

The graph saves its state after every step, so a claim stopped partway carries on from
its last step, and a claim waiting for a sign-off carries on once a handler signs off.
"""

import json
import os
from functools import cache
from typing import Annotated, Any, TypedDict

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, RetryPolicy, interrupt
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from agent import common
from agent.common import api, call, instructions, model, tracing
from agent.tools import TOOLS, steps

CALL_LIMIT = 30
POOL_SIZE = 4
SCHEMA = "undercover"
TOOL_NODE = ToolNode(TOOLS)


class ClaimState(TypedDict):
    """What the graph keeps for one claim."""

    messages: Annotated[list[AnyMessage], add_messages]
    claim_id: str
    model_calls: int
    decision: str | None
    payout: float
    status: str
    signed_off_by: str | None


def start_state(claim_id: str) -> ClaimState:
    """The state a claim starts with."""
    return {
        "messages": [HumanMessage(f"Please decide claim {claim_id}.")],
        "claim_id": claim_id,
        "model_calls": 0,
        "decision": None,
        "payout": 0.0,
        "status": "in_progress",
        "signed_off_by": None,
    }


@cache
def checkpointer() -> BaseCheckpointSaver[Any]:
    """Save progress in Supabase when DATABASE_URL is set, otherwise in memory."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return InMemorySaver()
    pool = ConnectionPool(
        database_url,
        max_size=POOL_SIZE,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "options": f"-c search_path={SCHEMA}",
        },
    )
    saver = PostgresSaver(pool)  # type: ignore[arg-type]
    saver.setup()
    return saver


def post(claim_id: str, message: AnyMessage) -> None:
    """Post a message from the claim agent as steps on the claim."""
    for kind, content in steps(message):
        call(
            "POST", f"/api/claims/{claim_id}/steps", {"kind": kind, "content": content}
        )


class Agent:
    """The agent node: the model with the tools and the claim instructions."""

    def __init__(self, llm: BaseChatModel, system_prompt: str) -> None:
        self.model = llm.bind_tools(TOOLS)
        self.system_message = SystemMessage(system_prompt)

    def __call__(self, state: ClaimState) -> dict[str, Any]:
        """Call the model with the conversation so far."""
        reply = self.model.invoke([self.system_message, *state["messages"]])
        post(state["claim_id"], reply)
        return {"messages": [reply], "model_calls": state["model_calls"] + 1}


def tools(state: ClaimState, config: RunnableConfig) -> dict[str, Any]:
    """Run the tools the model asked for, and keep the decision once it is recorded."""
    results = TOOL_NODE.invoke(state, config)["messages"]
    update: dict[str, Any] = {"messages": results}
    for message in results:
        post(state["claim_id"], message)
        if message.name == "record_decision" and not message.text.startswith("Refused"):
            claim = json.loads(message.text)
            update |= {key: claim[key] for key in ["decision", "payout", "status"]}
    return update


def signoff(state: ClaimState) -> dict[str, Any]:
    """Wait for a handler to sign off a payout over the limit."""
    answer = interrupt({"claim_id": state["claim_id"], "payout": state["payout"]})
    return {"signed_off_by": answer["handler"], "status": "in_progress"}


def pay(state: ClaimState) -> None:
    """Pay the approved payout; the backend counts a repeat as done."""
    claim_id = state["claim_id"]
    api("POST", f"/api/claims/{claim_id}/payment", {"amount": state["payout"]})
    call(
        "POST",
        f"/api/claims/{claim_id}/steps",
        {"kind": "message", "content": f"Paid ${state['payout']:,.2f}."},
    )


def after_agent(state: ClaimState) -> str:
    """Run the tools the model asked for, unless it stopped or used up its calls."""
    reply = state["messages"][-1]
    if isinstance(reply, AIMessage) and reply.tool_calls:
        return "tools" if state["model_calls"] < CALL_LIMIT else END
    return END


def after_tools(state: ClaimState) -> str:
    """Back to the model, or on from the decision as the backend recorded it."""
    if state["decision"] is None:
        return "agent"
    if state["status"] == "awaiting_signoff":
        return "signoff"
    if state["decision"] == "approve" and state["payout"] > 0:
        return "pay"
    return END


def build(llm: BaseChatModel, system_prompt: str) -> CompiledStateGraph[Any, Any, Any]:
    """Build the claim graph with its model, its instructions and the checkpointer."""
    model_retry = RetryPolicy(
        max_attempts=3,
        initial_interval=common.RETRY_DELAY,
        jitter=False,
        retry_on=Exception,
    )
    backend_retry = RetryPolicy(
        max_attempts=3,
        initial_interval=common.RETRY_DELAY,
        jitter=False,
        retry_on=httpx.TransportError,
    )
    builder = StateGraph(ClaimState)
    builder.add_node("agent", Agent(llm, system_prompt), retry_policy=model_retry)
    builder.add_node("tools", tools, retry_policy=backend_retry)
    builder.add_node("signoff", signoff)
    builder.add_node("pay", pay, retry_policy=backend_retry)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", after_agent, ["tools", END])
    builder.add_conditional_edges(
        "tools", after_tools, ["agent", "signoff", "pay", END]
    )
    builder.add_edge("signoff", "pay")
    builder.add_edge("pay", END)
    return builder.compile(checkpointer=checkpointer())


def thread(thread_id: str) -> RunnableConfig:
    """The config for a claim's thread, traced in its Langfuse session."""
    return tracing(thread_id) | {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 4 * CALL_LIMIT,
    }


def run(claim_id: str, thread_id: str, llm: BaseChatModel | None = None) -> None:
    """Work a claim on its thread: start it, carry it on, or resume it after sign-off.

    A claim still waiting for a sign-off, or a finished thread, is left as it is.
    """
    graph = build(llm or model(), instructions("claim"))
    config = thread(thread_id)
    saved = graph.get_state(config)
    command: Any
    if saved.interrupts:
        claim = api("GET", f"/api/claims/{claim_id}")
        if claim["signed_off_by"] is None:
            return
        command = Command(
            resume={"handler": claim["signed_off_by"], "at": claim["signed_off_at"]}
        )
    elif saved.next:
        command = None
    elif saved.values:
        return
    else:
        command = start_state(claim_id)
    graph.invoke(command, config, durability="sync")
