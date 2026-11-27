"""The UnderCover backend."""

import json
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import Connection, RowMapping

import agent.queue as runner
from backend import db
from backend.payout import Payout, PayoutRequest, Rules

INSTRUCTIONS = Path(__file__).parent / "instructions"
MAX_ATTEMPTS = 2
DOCUMENTS = {
    "policy-wording": ("policy_wording", "Policy wording"),
    "claims-manual": ("claims_manual", "Claims manual"),
}
DECISION_FIELDS = [
    "attempts",
    "run_id",
    "thread_id",
    "decision",
    "payout",
    "reason",
    "policy_section",
    "decided_at",
    "signed_off_by",
    "signed_off_at",
    "paid_amount",
    "paid_at",
]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Seed the database if it is missing, and close any run a restart cut short."""
    if not db.exists():
        db.seed()
    with db.engine().begin() as database:
        db.run(
            database,
            "UPDATE runs SET status = 'failed', problem = :problem, finished_at = :at "
            "WHERE status = 'running'",
            problem="The backend stopped during the run",
            at=db.now(),
        )
        for row in db.rows(
            database, "SELECT id FROM claims WHERE status = 'in_progress'"
        ):
            release(
                database, row["id"], "The backend stopped while the claim was worked"
            )
    yield


app = FastAPI(title="UnderCover", lifespan=lifespan)
Db = Annotated[Connection, Depends(db.connection)]


def find(database: Connection, table: str, record_id: str) -> RowMapping:
    """Return one row by id, or raise 404."""
    row = db.one(database, f"SELECT * FROM {table} WHERE id = :id", id=record_id)  # noqa: S608
    if row is None:
        raise HTTPException(404, f"{record_id} was not found")
    return row


def claim_view(row: RowMapping) -> dict[str, Any]:
    """Return a claim with its status, attempts, decision, sign-off and payment."""
    fields = {name: row[name] for name in DECISION_FIELDS}
    return json.loads(row["record"]) | {"status": row["status"]} | fields


@app.get("/api/health", operation_id="health")
def health() -> dict[str, str]:
    """Check that the backend is running."""
    return {"status": "ok"}


@app.get("/api/claims", operation_id="list_claims")
def list_claims(database: Db) -> list[dict[str, Any]]:
    """List the claims queue with each claim's status and attempts."""
    keys = ["claim_id", "claim_type", "summary", "amount_claimed", "submitted_on"]
    claims = []
    for row in db.rows(database, "SELECT * FROM claims ORDER BY id"):
        record = json.loads(row["record"])
        claims.append(
            {key: record[key] for key in keys}
            | {
                "status": row["status"],
                "decision": row["decision"],
                "attempts": row["attempts"],
            }
        )
    return claims


@app.get("/api/claims/{claim_id}", operation_id="get_claim")
def get_claim(claim_id: str, database: Db) -> dict[str, Any]:
    """Get a claim in full, with its decision, sign-off and payment."""
    return claim_view(find(database, "claims", claim_id))


@app.get("/api/policies/{policy_id}", operation_id="get_policy")
def get_policy(policy_id: str, database: Db) -> dict[str, Any]:
    """Get a policy with its plan's limits and the details of its add-ons."""
    policy = json.loads(find(database, "policies", policy_id)["record"])
    plans = json.loads(db.setting(database, "plans"))
    plan = next(p for p in plans["plans"] if p["plan"] == policy["plan"])
    add_ons = [a for a in plans["add_ons"] if a["add_on"] in policy["add_ons"]]
    return policy | {
        "plan_details": plan,
        "deductible_applies_to": plans["deductible_applies_to"],
        "add_on_details": add_ons,
    }


@app.post("/api/policies/{policy_id}/payout", operation_id="calculate_payout")
def calculate_payout(policy_id: str, request: PayoutRequest, database: Db) -> Payout:
    """Work out the payout on an approved claim with UnderCover's payout rules."""
    policy = json.loads(find(database, "policies", policy_id)["record"])
    plans = json.loads(db.setting(database, "plans"))
    company = json.loads(db.setting(database, "company"))
    return Rules(policy, plans, company).work_out(request)


@app.get("/api/customers/{customer_id}/history", operation_id="get_claim_history")
def get_claim_history(customer_id: str, database: Db) -> dict[str, Any]:
    """Get a customer's earlier claims, without their personal details."""
    customer = json.loads(find(database, "customers", customer_id)["record"])
    return {
        "customer_id": customer_id,
        "customer_since": customer["customer_since"],
        "previous_claims": customer["previous_claims"],
    }


@app.get("/api/policy-wording", operation_id="get_policy_wording")
def get_policy_wording(database: Db) -> dict[str, str]:
    """Get the policy wording that decisions follow."""
    return {"wording": db.setting(database, "policy_wording")}


def document_parts(database: Connection, document: str) -> tuple[str, str, list[str]]:
    """Return a document's title, introduction and sections, or raise 404."""
    if document not in DOCUMENTS:
        raise HTTPException(
            404, f"{document} is not a document: use policy-wording or claims-manual"
        )
    setting, title = DOCUMENTS[document]
    introduction, *sections = db.setting(database, setting).split("\n## ")
    return title, introduction.strip(), [section.strip() for section in sections]


@app.get("/api/documents/{document}/sections", operation_id="list_sections")
def list_sections(document: str, database: Db) -> dict[str, Any]:
    """List the numbered sections of the policy wording or the claims manual."""
    title, introduction, sections = document_parts(database, document)
    return {
        "document": title,
        "introduction": introduction,
        "sections": [section.splitlines()[0] for section in sections],
    }


@app.get("/api/documents/{document}/sections/{number}", operation_id="read_section")
def read_section(document: str, number: str, database: Db) -> dict[str, str]:
    """Read one numbered section of the policy wording or the claims manual."""
    title, _, sections = document_parts(database, document)
    for section in sections:
        heading, _, text = section.partition("\n")
        if heading.startswith(f"{number}."):
            return {"document": title, "section": heading, "text": text.strip()}
    raise HTTPException(404, f"The {title.lower()} has no section {number}")


@app.get("/api/instructions", operation_id="get_instructions")
def get_instructions() -> dict[str, str]:
    """Get the agent's instructions for running the queue and for working a claim."""
    return {
        name: (INSTRUCTIONS / f"{name}.md").read_text(encoding="utf-8")
        for name in ["queue", "claim"]
    }


@app.get("/api/handlers", operation_id="list_handlers")
def list_handlers(database: Db) -> list[dict[str, Any]]:
    """List the claims handlers and whether each can sign off large payouts."""
    keys = ["handler_id", "name", "role", "can_approve_over_limit"]
    return [
        {key: json.loads(row["record"])[key] for key in keys}
        for row in db.rows(database, "SELECT record FROM handlers ORDER BY id")
    ]


class Decision(BaseModel):
    """The agent's decision on a claim."""

    decision: Literal["approve", "decline", "refer"]
    payout: float = Field(ge=0)
    reason: str
    policy_section: str

    @model_validator(mode="after")
    def only_approvals_pay(self) -> Self:
        """Check that a decline or a referral pays nothing."""
        if self.decision != "approve" and self.payout != 0:
            raise ValueError("A decline or a referral has a payout of 0")
        return self


class Payment(BaseModel):
    """A payment of an approved claim."""

    amount: float


class SignOff(BaseModel):
    """A handler's sign-off of a large payout."""

    handler_id: str


class StartClaim(BaseModel):
    """The run that starts work on a claim."""

    run_id: int


class Step(BaseModel):
    """One step the agent took on a claim."""

    kind: Literal["tool_call", "tool_result", "message", "read", "attempt", "error"]
    content: str


def approval_limit(database: Connection) -> float:
    """Return the payout above which a handler must sign off."""
    return json.loads(db.setting(database, "company"))["payout_approval_limit"]


def require_status(row: RowMapping, status: str) -> None:
    """Raise 409 unless the claim is in the given status."""
    if row["status"] != status:
        raise HTTPException(409, f"{row['id']} is {row['status']}, not {status}")


def add_step(database: Connection, claim_id: str, kind: str, text: str) -> None:
    """Record one step on a claim."""
    db.run(
        database,
        "INSERT INTO steps (claim_id, kind, content, created_at) "
        "VALUES (:claim_id, :kind, :content, :at)",
        claim_id=claim_id,
        kind=kind,
        content=text,
        at=db.now(),
    )


class RunRequest(BaseModel):
    """How many claims a run of the queue may work."""

    cap: int = Field(ge=1)


class Todo(BaseModel):
    """One item on a run's to-do list."""

    claim_id: str
    content: str
    status: Literal["todo", "in_progress", "done", "handler"]


class Todos(BaseModel):
    """A run's to-do list."""

    todos: list[Todo]


class Finish(BaseModel):
    """The summary of a finished run, for the handlers."""

    summary: str


class Problem(BaseModel):
    """What went wrong."""

    problem: str


def run_view(row: RowMapping) -> dict[str, Any]:
    """Return a run with its to-do list."""
    return dict(row) | {"todos": json.loads(row["todos"])}


def find_run(database: Connection, run_id: int) -> RowMapping:
    """Return a run by id, or raise 404."""
    row = db.one(database, "SELECT * FROM runs WHERE id = :id", id=run_id)
    if row is None:
        raise HTTPException(404, f"Run {run_id} was not found")
    return row


def require_running(database: Connection, run_id: int) -> None:
    """Raise 409 unless the run is still running."""
    status = find_run(database, run_id)["status"]
    if status != "running":
        raise HTTPException(409, f"Run {run_id} is {status}, not running")


def begin_run(
    database: Connection,
    background: BackgroundTasks,
    claim_id: str | None,
    cap: int,
) -> dict[str, Any]:
    """Create a run and start the agent on it, unless a run is already going."""
    if db.one(database, "SELECT 1 FROM runs WHERE status = 'running'"):
        raise HTTPException(409, "A run is already going; wait for it to finish")
    run_id = db.scalar(
        database,
        "INSERT INTO runs (claim_id, cap, started_at) "
        "VALUES (:claim_id, :cap, :at) RETURNING id",
        claim_id=claim_id,
        cap=cap,
        at=db.now(),
    )
    database.commit()
    row = find_run(database, run_id)
    if os.getenv("UNDERCOVER_AGENT") != "off":
        background.add_task(runner.run_queue, row["id"])
    return run_view(row)


@app.post("/api/runs", operation_id="start_run")
def start_run(
    request: RunRequest, database: Db, background: BackgroundTasks
) -> dict[str, Any]:
    """Start the UnderCover Agent on the waiting claims, up to the cap."""
    return begin_run(database, background, None, request.cap)


@app.get("/api/runs/latest", operation_id="get_latest_run")
def get_latest_run(database: Db) -> dict[str, Any] | None:
    """Get the latest run with its to-do list, or null before the first run."""
    row = db.one(database, "SELECT * FROM runs ORDER BY id DESC LIMIT 1")
    return None if row is None else run_view(row)


@app.get("/api/runs/{run_id}", operation_id="get_run")
def get_run(run_id: int, database: Db) -> dict[str, Any]:
    """Get a run with its scope, cap and to-do list."""
    return run_view(find_run(database, run_id))


@app.post("/api/runs/{run_id}/todos", operation_id="write_todos")
def write_todos(run_id: int, todos: Todos, database: Db) -> dict[str, Any]:
    """Replace the to-do list of a run that is still running."""
    require_running(database, run_id)
    items = [todo.model_dump() for todo in todos.todos]
    db.run(
        database,
        "UPDATE runs SET todos = :todos WHERE id = :id",
        todos=json.dumps(items),
        id=run_id,
    )
    return run_view(find_run(database, run_id))


@app.post("/api/runs/{run_id}/finish", operation_id="finish_run")
def finish_run(run_id: int, finish: Finish, database: Db) -> dict[str, Any]:
    """Finish a run with a summary for the handlers."""
    require_running(database, run_id)
    db.run(
        database,
        "UPDATE runs SET status = 'finished', summary = :summary, finished_at = :at "
        "WHERE id = :id",
        summary=finish.summary,
        at=db.now(),
        id=run_id,
    )
    return run_view(find_run(database, run_id))


@app.post("/api/runs/{run_id}/fail", operation_id="fail_run")
def fail_run(run_id: int, failure: Problem, database: Db) -> dict[str, Any]:
    """Mark a run as failed, with the problem."""
    require_running(database, run_id)
    db.run(
        database,
        "UPDATE runs SET status = 'failed', problem = :problem, finished_at = :at "
        "WHERE id = :id",
        problem=failure.problem,
        at=db.now(),
        id=run_id,
    )
    return run_view(find_run(database, run_id))


@app.post("/api/claims/{claim_id}/run", operation_id="run_agent")
def run_agent(
    claim_id: str, database: Db, background: BackgroundTasks
) -> dict[str, Any]:
    """Start the UnderCover Agent on one waiting claim."""
    require_status(find(database, "claims", claim_id), "waiting")
    return begin_run(database, background, claim_id, 1)


@app.post("/api/claims/{claim_id}/start", operation_id="start_claim")
def start_claim(claim_id: str, start: StartClaim, database: Db) -> dict[str, Any]:
    """Start a waiting claim in a run, on a new thread for this attempt."""
    row = find(database, "claims", claim_id)
    require_status(row, "waiting")
    if row["attempts"]:
        add_step(database, claim_id, "attempt", f"Attempt {row['attempts'] + 1}")
    db.run(
        database,
        "UPDATE claims SET status = 'in_progress', run_id = :run_id, "
        "thread_id = :thread_id WHERE id = :id",
        run_id=start.run_id,
        thread_id=f"{claim_id}-{uuid.uuid4().hex[:8]}",
        id=claim_id,
    )
    return claim_view(find(database, "claims", claim_id))


def release(database: Connection, claim_id: str, problem: str) -> None:
    """Record a failed attempt: back to waiting, or left for a handler at the limit."""
    attempts = find(database, "claims", claim_id)["attempts"] + 1
    add_step(database, claim_id, "error", problem)
    db.run(
        database,
        "UPDATE claims SET status = :status, attempts = :attempts WHERE id = :id",
        status="waiting" if attempts < MAX_ATTEMPTS else "needs_handler",
        attempts=attempts,
        id=claim_id,
    )


@app.post("/api/claims/{claim_id}/release", operation_id="release_claim")
def release_claim(claim_id: str, failure: Problem, database: Db) -> dict[str, Any]:
    """Record a failed attempt on a claim; a second failure leaves it for a handler."""
    row = find(database, "claims", claim_id)
    require_status(row, "in_progress")
    if row["decision"] is not None:
        raise HTTPException(409, f"{claim_id} is decided, so it stays as it is")
    release(database, claim_id, failure.problem)
    return claim_view(find(database, "claims", claim_id))


@app.post("/api/claims/{claim_id}/steps", operation_id="record_step")
def record_step(claim_id: str, step: Step, database: Db) -> dict[str, str]:
    """Record a step the agent took on a claim."""
    find(database, "claims", claim_id)
    add_step(database, claim_id, step.kind, step.content)
    return {"status": "recorded"}


@app.get("/api/claims/{claim_id}/steps", operation_id="list_steps")
def list_steps(claim_id: str, database: Db) -> list[dict[str, Any]]:
    """List the steps the agent took on a claim, in order."""
    find(database, "claims", claim_id)
    return [
        dict(row)
        for row in db.rows(
            database,
            "SELECT kind, content, created_at FROM steps WHERE claim_id = :id "
            "ORDER BY id",
            id=claim_id,
        )
    ]


@app.post("/api/claims/{claim_id}/decision", operation_id="record_decision")
def record_decision(claim_id: str, decision: Decision, database: Db) -> dict[str, Any]:
    """Record the decision on a claim in progress; a repeat of it counts as done."""
    row = find(database, "claims", claim_id)
    if row["decision"] is not None:
        if (row["decision"], row["payout"]) == (decision.decision, decision.payout):
            return claim_view(row)
        raise HTTPException(
            409, f"{claim_id} is already decided ({row['decision']}), and that stands"
        )
    require_status(row, "in_progress")
    over_limit = decision.payout > approval_limit(database)
    db.run(
        database,
        "UPDATE claims SET status = :status, decision = :decision, payout = :payout, "
        "reason = :reason, policy_section = :section, decided_at = :at WHERE id = :id",
        status="awaiting_signoff" if over_limit else "decided",
        decision=decision.decision,
        payout=decision.payout,
        reason=decision.reason,
        section=decision.policy_section,
        at=db.now(),
        id=claim_id,
    )
    return claim_view(find(database, "claims", claim_id))


def payment_problem(row: RowMapping, amount: float, limit: float) -> str | None:
    """Return why a payment can't be made, or None if it can."""
    if row["decision"] is None:
        return f"{row['id']} has no decision yet"
    if row["decision"] != "approve":
        return f"{row['id']} was not approved, so nothing is paid"
    if row["paid_at"] is not None:
        return f"{row['id']} has already been paid"
    if row["payout"] > limit and row["signed_off_by"] is None:
        return (
            f"The payout is over ${limit:,.2f}, so the payment waits for a "
            "handler's sign-off"
        )
    if amount != row["payout"]:
        return f"The amount must be the approved payout of ${row['payout']:,.2f}"
    return None


@app.post("/api/claims/{claim_id}/payment", operation_id="pay_claim")
def pay_claim(claim_id: str, payment: Payment, database: Db) -> dict[str, Any]:
    """Pay the approved payout on a claim; a repeat of the payment counts as done."""
    row = find(database, "claims", claim_id)
    if row["paid_at"] is not None and row["paid_amount"] == payment.amount:
        return claim_view(row)
    problem = payment_problem(row, payment.amount, approval_limit(database))
    if problem:
        raise HTTPException(409, problem)
    db.run(
        database,
        "UPDATE claims SET status = 'decided', paid_amount = :amount, paid_at = :at "
        "WHERE id = :id",
        amount=payment.amount,
        at=db.now(),
        id=claim_id,
    )
    return claim_view(find(database, "claims", claim_id))


@app.post("/api/claims/{claim_id}/signoff", operation_id="sign_off")
def sign_off(claim_id: str, signoff: SignOff, database: Db) -> dict[str, Any]:
    """Sign off a payout over the approval limit; the agent then pays it."""
    require_status(find(database, "claims", claim_id), "awaiting_signoff")
    handler = json.loads(find(database, "handlers", signoff.handler_id)["record"])
    if not handler["can_approve_over_limit"]:
        raise HTTPException(403, f"{handler['name']} can't sign off large payouts")
    db.run(
        database,
        "UPDATE claims SET status = 'in_progress', signed_off_by = :handler, "
        "signed_off_at = :at WHERE id = :id",
        handler=signoff.handler_id,
        at=db.now(),
        id=claim_id,
    )
    return claim_view(find(database, "claims", claim_id))
