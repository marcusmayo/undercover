from pathlib import Path

import httpx
import pytest

from backend import main

RUN = {"run_id": 1}

TODOS = [
    {"claim_id": "CLM-1001", "content": "Flight canceled", "status": "done"},
    {"claim_id": "CLM-1002", "content": "Lost bag", "status": "in_progress"},
]


@pytest.fixture
def started(client, told):
    """The runs handed to the agent's service."""
    return told


def test_no_run_yet(client):
    assert client.get("/api/runs/latest").json() is None


def test_start_run(client, started):
    run = client.post("/api/runs", json={"cap": 5}).json()
    assert run["status"] == "running"
    assert run["cap"] == 5
    assert run["claim_id"] is None
    assert run["todos"] == []
    assert run["started_at"]
    assert started == [f"/runs/{run['id']}"]
    assert client.get("/api/runs/latest").json() == run
    assert client.get(f"/api/runs/{run['id']}").json() == run


def test_one_run_at_a_time(client, started):
    client.post("/api/runs", json={"cap": 5})
    response = client.post("/api/runs", json={"cap": 5})
    assert response.status_code == 409
    assert response.json()["detail"] == "A run is already going; wait for it to finish"
    assert client.post("/api/claims/CLM-1001/run").status_code == 409
    assert len(started) == 1


@pytest.mark.parametrize("cap", [0, -1])
def test_cap_below_one(client, started, cap):
    assert client.post("/api/runs", json={"cap": cap}).status_code == 422
    assert started == []


def test_unknown_run(client):
    response = client.get("/api/runs/99")
    assert response.status_code == 404
    assert response.json()["detail"] == "Run 99 was not found"


def test_write_todos(client, started):
    run_id = client.post("/api/runs", json={"cap": 2}).json()["id"]
    run = client.post(f"/api/runs/{run_id}/todos", json={"todos": TODOS}).json()
    assert run["todos"] == TODOS
    assert client.get("/api/runs/latest").json()["todos"] == TODOS


def test_todo_with_an_unknown_status(client, started):
    run_id = client.post("/api/runs", json={"cap": 2}).json()["id"]
    bad = [TODOS[0] | {"status": "maybe"}]
    response = client.post(f"/api/runs/{run_id}/todos", json={"todos": bad})
    assert response.status_code == 422


def test_finish_run(client, started):
    run_id = client.post("/api/runs", json={"cap": 2}).json()["id"]
    run = client.post(
        f"/api/runs/{run_id}/finish", json={"summary": "Two claims decided."}
    ).json()
    assert run["status"] == "finished"
    assert run["summary"] == "Two claims decided."
    assert run["finished_at"]
    response = client.post(f"/api/runs/{run_id}/todos", json={"todos": TODOS})
    assert response.status_code == 409
    assert response.json()["detail"] == f"Run {run_id} is finished, not running"
    assert client.post("/api/runs", json={"cap": 1}).status_code == 200


def test_fail_run(client, started):
    run_id = client.post("/api/runs", json={"cap": 2}).json()["id"]
    run = client.post(
        f"/api/runs/{run_id}/fail", json={"problem": "The model provider is down"}
    ).json()
    assert run["status"] == "failed"
    assert run["problem"] == "The model provider is down"
    assert run["finished_at"]
    response = client.post(f"/api/runs/{run_id}/finish", json={"summary": "Done."})
    assert response.status_code == 409


def test_run_agent_starts_a_one_claim_run(client, started):
    run = client.post("/api/claims/CLM-1001/run").json()
    assert run["claim_id"] == "CLM-1001"
    assert run["cap"] == 1
    assert started == [f"/runs/{run['id']}"]


def test_run_agent_needs_a_waiting_claim(client, started):
    client.post("/api/claims/CLM-1001/start", json=RUN)
    assert client.post("/api/claims/CLM-1001/run").status_code == 409
    assert started == []


def test_a_signoff_is_handed_to_the_agent_once_saved(client, told):
    client.post("/api/claims/CLM-1003/start", json=RUN)
    client.post(
        "/api/claims/CLM-1003/decision",
        json={
            "decision": "approve",
            "payout": 3750.0,
            "reason": "Covered treatment abroad.",
            "policy_section": "7. Emergency Medical",
        },
    )
    assert told == []
    client.post("/api/claims/CLM-1003/signoff", json={"handler_id": "HND-01"})
    assert told == ["/claims/CLM-1003/resume"]


def test_tell_agent_posts_to_the_agents_service(monkeypatch):
    posted = []
    monkeypatch.setattr(httpx, "post", lambda url, timeout: posted.append(url))
    monkeypatch.setenv("UNDERCOVER_AGENT", "http://127.0.0.1:8003")
    main.tell_agent("/runs/1")
    monkeypatch.delenv("UNDERCOVER_AGENT")
    main.tell_agent("/claims/CLM-1001/resume")
    assert posted == [
        "http://127.0.0.1:8003/runs/1",
        "http://127.0.0.1:8002/claims/CLM-1001/resume",
    ]


def test_a_signoff_stands_when_the_agent_cant_be_reached(monkeypatch):
    def unreachable(url, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", unreachable)
    main.tell_agent("/claims/CLM-1001/resume")


def test_the_backend_does_not_import_the_agent():
    source = Path(main.__file__).read_text(encoding="utf-8")
    assert "import agent" not in source
    assert "from agent" not in source
