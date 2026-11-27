"""The claim agent: a fresh agent for each claim, which decides it and reports back."""

import json
from collections.abc import Iterator
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent.common import api, call, instructions, retries, tracing

CALL_LIMIT = 25
STEP_LENGTH = 400
Document = Literal["policy-wording", "claims-manual"]
Benefit = Literal[
    "trip_cancellation",
    "trip_interruption",
    "trip_delay",
    "baggage_delay",
    "baggage",
    "passport",
    "emergency_medical",
]


class Line(BaseModel):
    """One covered item or expense on a claim."""

    description: str = Field(description="What the item or expense is")
    amount: float = Field(
        description="The total claimed for the line in US dollars, before any "
        "limits. For several items, the total for all of them"
    )
    quantity: int = Field(
        default=1,
        ge=1,
        description="How many separate items the line holds, such as 3 for three "
        "shirts. A pair or a set is one item",
    )
    kind: Literal[
        "item", "electronics", "cash", "expense", "hotel", "meals_without_receipts"
    ] = Field(
        description="item: a belonging, or a pair or set that counts as one item. "
        "electronics: only a phone, tablet, laptop, camera or its lens, e-reader or "
        "smartwatch; any other device is an item. cash: cash that was lost or "
        "stolen. expense: any other cost with a receipt, such as a medical bill, a "
        "meal, a taxi, a fee or a prepaid trip cost. hotel: a hotel during a trip "
        "delay. meals_without_receipts: meals during a trip delay with no receipts."
    )
    nights: int = Field(default=0, description="For a hotel, the number of nights")
    hours: float = Field(
        default=0,
        description="For meals without receipts, the length of the delay in hours",
    )


@tool
def get_claim(claim_id: str) -> str:
    """Get a claim in full: the story, the items claimed and the documents."""
    return call("GET", f"/api/claims/{claim_id}")


@tool
def get_policy(policy_id: str) -> str:
    """Get a policy with its plan's limits and the details of its add-ons."""
    return call("GET", f"/api/policies/{policy_id}")


@tool
def get_claim_history(customer_id: str) -> str:
    """Get a customer's earlier claims."""
    return call("GET", f"/api/customers/{customer_id}/history")


@tool
def list_sections(document: Document) -> str:
    """List the numbered sections of the policy wording or the claims manual."""
    return call("GET", f"/api/documents/{document}/sections")


@tool
def read_section(document: Document, number: str) -> str:
    """Read one numbered section of the policy wording or the claims manual."""
    return call("GET", f"/api/documents/{document}/sections/{number}")


@tool(parse_docstring=True)
def calculate_payout(
    policy_id: str, benefit: Benefit, lines: list[Line], paid_by_others: float
) -> str:
    """Work out the payout on an approved claim with UnderCover's payout rules.

    Args:
        policy_id: The customer's policy id
        benefit: The benefit the claim is paid under
        lines: The covered items and expenses, one for each line of the claim, with
            any excluded ones left out. A pair or a set counts as one item
        paid_by_others: Anything a carrier, hotel or other party has paid or
            refunded for the same loss, or 0
    """
    body = {
        "benefit": benefit,
        "lines": [Line.model_validate(line).model_dump() for line in lines],
        "paid_by_others": paid_by_others,
    }
    return call("POST", f"/api/policies/{policy_id}/payout", body)


@tool
def record_decision(
    claim_id: str,
    decision: Literal["approve", "decline", "refer"],
    payout: float,
    reason: str,
    policy_section: str,
) -> str:
    """Record the decision on a claim, with the payout, the reason and the section."""
    body = {
        "decision": decision,
        "payout": payout,
        "reason": reason,
        "policy_section": policy_section,
    }
    return call("POST", f"/api/claims/{claim_id}/decision", body)


@tool
def pay_claim(claim_id: str, amount: float) -> str:
    """Pay the approved payout on a claim."""
    return call("POST", f"/api/claims/{claim_id}/payment", {"amount": amount})


TOOLS = [
    get_claim,
    get_policy,
    get_claim_history,
    list_sections,
    read_section,
    calculate_payout,
    record_decision,
    pay_claim,
]


def shorten(text: str) -> str:
    """Shorten long text for a step."""
    return text if len(text) <= STEP_LENGTH else text[:STEP_LENGTH] + "..."


def steps(message: BaseMessage) -> Iterator[tuple[str, str]]:
    """Describe a message from the claim agent as steps for the dashboard."""
    if isinstance(message, ToolMessage):
        yield "tool_result", f"{message.name}: {shorten(message.text)}"
        if message.name == "read_section" and not message.text.startswith("Refused"):
            section = json.loads(message.text)
            yield "read", f"{section['document']}, {section['section']}"
    elif isinstance(message, AIMessage):
        if message.text:
            yield "message", message.text
        for call_ in message.tool_calls:
            yield "tool_call", f"{call_['name']}({json.dumps(call_['args'])})"


def decide(claim_id: str, thread_id: str, llm: BaseChatModel) -> None:
    """Let a fresh claim agent work a started claim, posting each step to the claim."""
    limit: AgentMiddleware[Any, Any, Any] = ModelCallLimitMiddleware(
        run_limit=CALL_LIMIT
    )
    agent = create_agent(
        llm,
        tools=TOOLS,
        system_prompt=instructions("claim"),
        middleware=[*retries(), limit],
    )
    for update in agent.stream(
        {"messages": [{"role": "user", "content": f"Please decide claim {claim_id}."}]},
        config=tracing(thread_id) | {"recursion_limit": 4 * CALL_LIMIT},
        stream_mode="updates",
    ):
        for node in update.values():
            for message in (node or {}).get("messages", []):
                for kind, content in steps(message):
                    call(
                        "POST",
                        f"/api/claims/{claim_id}/steps",
                        {"kind": kind, "content": content},
                    )


def outcome(claim: dict[str, Any]) -> str:
    """Describe a decided claim in one line."""
    line = f"{claim['claim_id']}: {claim['decision']}"
    if claim["decision"] != "approve":
        return line
    if claim["paid_at"]:
        paid = "paid"
    elif claim["status"] == "awaiting_signoff":
        paid = "waiting for a handler's sign-off"
    else:
        paid = "not paid yet"
    return f"{line}, ${claim['payout']:,.2f}, {paid}"


def work_claim(run_id: int, claim_id: str, llm: BaseChatModel) -> str:
    """Work one waiting claim with a fresh claim agent and report the outcome in a line.

    It never raises: a claim that fails before it is decided is released, which puts
    it back in the queue or, on its second failure, leaves it for a handler.
    """
    started = call("POST", f"/api/claims/{claim_id}/start", {"run_id": run_id})
    if started.startswith("Refused"):
        return started
    try:
        decide(claim_id, json.loads(started)["thread_id"], llm)
        problem = "the agent finished without recording a decision"
    except Exception as error:
        problem = f"the agent stopped with an error: {error}"
    claim = api("GET", f"/api/claims/{claim_id}")
    if claim["status"] != "in_progress":
        return outcome(claim)
    claim = api(
        "POST", f"/api/claims/{claim_id}/release", {"problem": problem.capitalize()}
    )
    where = (
        "back in the queue" if claim["status"] == "waiting" else "left for a handler"
    )
    return f"{claim_id}: failed ({problem}), {where}"
