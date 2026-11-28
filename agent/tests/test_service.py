import pytest
from fastapi.testclient import TestClient

from agent import claim, graph, queue, service
from agent.tests.conftest import decided, decision, stub, tool_call

agent_service = TestClient(service.app)


@pytest.fixture
def held(client, monkeypatch):
    """CLM-1003 approved for $3,750.00 and waiting for a sign-off."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    claim.work_claim(1, "CLM-1003", stub(decision("CLM-1003", 3750.0)))
    return client


def test_health():
    assert agent_service.get("/health").json() == {"status": "ok"}


def test_work_run_works_the_run_in_the_background(monkeypatch):
    runs = []
    monkeypatch.setattr(queue, "run_queue", runs.append)
    assert agent_service.post("/runs/7").json() == {"status": "started"}
    assert runs == [7]


def test_resume_claim_pays_a_signed_off_claim_once(held):
    held.post("/api/claims/CLM-1003/signoff", json={"handler_id": "HND-01"})
    response = agent_service.post("/claims/CLM-1003/resume")
    assert response.json() == {"status": "resuming"}
    paid = held.get("/api/claims/CLM-1003").json()
    assert paid["status"] == "decided"
    assert paid["paid_amount"] == 3750.0
    agent_service.post("/claims/CLM-1003/resume")
    steps = [s["content"] for s in held.get("/api/claims/CLM-1003/steps").json()]
    assert steps.count("Paid $3,750.00.") == 1


def test_resume_before_a_signoff_changes_nothing(held):
    agent_service.post("/claims/CLM-1003/resume")
    assert held.get("/api/claims/CLM-1003").json()["status"] == "awaiting_signoff"


def test_catching_up_carries_on_stopped_and_signed_off_claims(held, monkeypatch):
    with pytest.raises(KeyboardInterrupt):
        claim.work_claim(
            1,
            "CLM-1001",
            stub(tool_call("get_claim", claim_id="CLM-1001"), KeyboardInterrupt()),
        )
    held.post("/api/claims/CLM-1003/signoff", json={"handler_id": "HND-01"})
    monkeypatch.setattr(service, "model", lambda: stub(*decided("CLM-1001")))
    service.catch_up()
    first = held.get("/api/claims/CLM-1001").json()
    assert first["status"] == "decided"
    assert first["attempts"] == 0
    assert held.get("/api/claims/CLM-1003").json()["paid_amount"] == 3750.0


def test_catching_up_carries_on_a_stopped_run(client, monkeypatch):
    runs = []
    monkeypatch.setattr(queue, "run_queue", runs.append)
    run_id = client.post("/api/runs", json={"cap": 1}).json()["id"]
    service.catch_up()
    assert runs == [run_id]
    client.post(f"/api/runs/{run_id}/finish", json={"summary": "Done."})
    monkeypatch.setattr(service, "model", stub)
    service.catch_up()
    assert runs == [run_id]


def test_the_service_starts_catching_up_when_it_starts(monkeypatch):
    started = []

    class Thread:
        def __init__(self, target, daemon):
            started.append((target, daemon))

        def start(self):
            started.append("started")

    monkeypatch.setattr(service.threading, "Thread", Thread)
    with TestClient(service.app):
        pass
    assert started == [(service.catch_up, True), "started"]


def test_the_entry_point_takes_a_claim_and_a_thread():
    assert list(graph.run.__annotations__)[:2] == ["claim_id", "thread_id"]
