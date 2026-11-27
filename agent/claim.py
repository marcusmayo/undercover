"""Working one claim with the claim graph, and reporting the outcome in a line."""

import json
from typing import Any

from langchain_core.language_models import BaseChatModel

from agent import graph
from agent.common import api, call


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


def carry_on(claim_id: str, thread_id: str, llm: BaseChatModel | None) -> str:
    """Run the claim graph on a claim's thread and report the outcome in a line.

    It never raises on an error: a claim that fails before it is decided is released,
    which puts it back in the queue or, on its second failure, leaves it for a handler.
    A stop, such as Ctrl-C, is not an error: the claim keeps its thread to carry on.
    """
    try:
        graph.run(claim_id, thread_id, llm)
        problem = "the agent finished without recording a decision"
    except Exception as error:
        problem = f"the agent stopped with an error: {error}"
    claim = api("GET", f"/api/claims/{claim_id}")
    if claim["decision"] is not None or claim["status"] != "in_progress":
        return outcome(claim)
    claim = api(
        "POST", f"/api/claims/{claim_id}/release", {"problem": problem.capitalize()}
    )
    where = (
        "back in the queue" if claim["status"] == "waiting" else "left for a handler"
    )
    return f"{claim_id}: failed ({problem}), {where}"


def work_claim(run_id: int, claim_id: str, llm: BaseChatModel | None) -> str:
    """Start a waiting claim in a run on a new thread, and work it to its outcome."""
    started = call("POST", f"/api/claims/{claim_id}/start", {"run_id": run_id})
    if started.startswith("Refused"):
        return started
    return carry_on(claim_id, json.loads(started)["thread_id"], llm)
