"""The agent's service: it works runs and picks claims back up after a sign-off.

`npm run dev` starts it on port 8002. When it starts, it carries on whatever was left
while it was away: a run that was stopped, a claim stopped partway, a claim signed off.
"""

import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI

from agent import claim, queue
from agent.common import api, model


def catch_up() -> None:
    """Carry on a run that was stopped, or else every claim left in progress."""
    latest = api("GET", "/api/runs/latest")
    if latest and latest["status"] == "running":
        queue.run_queue(latest["id"])
    else:
        claim.carry_on_all(model())


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Catch up in the background once the service is up."""
    threading.Thread(target=catch_up, daemon=True).start()
    yield


app = FastAPI(title="UnderCover Agent", lifespan=lifespan)


@app.get("/health", operation_id="health")
def health() -> dict[str, str]:
    """Check that the agent's service is running."""
    return {"status": "ok"}


@app.post("/runs/{run_id}", operation_id="work_run")
def work_run(run_id: int, background: BackgroundTasks) -> dict[str, str]:
    """Work a run that the backend has started, in the background."""
    background.add_task(queue.run_queue, run_id)
    return {"status": "started"}


@app.post("/claims/{claim_id}/resume", operation_id="resume_claim")
def resume_claim(claim_id: str, background: BackgroundTasks) -> dict[str, str]:
    """Pick a claim back up after a handler's sign-off, in the background."""
    background.add_task(claim.carry_on, claim_id, None)
    return {"status": "resuming"}
