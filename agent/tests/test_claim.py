import inspect
import json

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from agent import claim, common, graph, tools
from agent.tests.conftest import DOWN, decided, decision, stub, tool_call
from backend import db

READS = [
    tool_call("get_claim", claim_id="CLM-1001"),
    tool_call("get_policy", policy_id="POL-2001"),
    tool_call("get_claim_history", customer_id="CUS-3001"),
    tool_call("list_sections", document="policy-wording"),
    tool_call("read_section", document="policy-wording", number="3"),
    tool_call("read_section", document="claims-manual", number="9"),
]
PAYOUT = tool_call(
    "calculate_payout",
    policy_id="POL-2001",
    benefit="trip_delay",
    lines=[
        {"description": "Hotel", "amount": 300.0, "kind": "hotel", "nights": 1},
        {"description": "Meals", "amount": 120.0, "kind": "expense"},
    ],
    paid_by_others=0,
)
DECLINE = tool_call(
    "record_decision",
    claim_id="CLM-1001",
    decision="decline",
    payout=0,
    reason="Not covered.",
    policy_section="9. General exclusions",
)


def get(client, claim_id="CLM-1001"):
    return client.get(f"/api/claims/{claim_id}").json()


def steps(client, kind=None):
    found = client.get("/api/claims/CLM-1001/steps").json()
    return [s["content"] for s in found if kind in (None, s["kind"])]


def work(llm, claim_id="CLM-1001"):
    return claim.work_claim(1, claim_id, llm)


def saved(claim_id="CLM-1001"):
    """The claim thread's saved state."""
    return graph.build(stub(), "").get_state(graph.thread(get_thread(claim_id)))


def get_thread(claim_id):
    with db.engine().connect() as database:
        return db.scalar(
            database, "SELECT thread_id FROM claims WHERE id = :id", id=claim_id
        )


def set_limit(limit):
    with db.engine().begin() as database:
        company = json.loads(db.setting(database, "company"))
        company["payout_approval_limit"] = limit
        db.run(
            database,
            "UPDATE settings SET content = :content WHERE name = 'company'",
            content=json.dumps(company),
        )


def test_full_claim(client):
    llm = stub(*READS, PAYOUT, decision())
    assert work(llm) == "CLM-1001: approve, $420.00, paid"
    paid = get(client)
    assert paid["paid_amount"] == 420.0
    assert paid["status"] == "decided"
    calls = [s.split("(")[0] for s in steps(client, "tool_call")]
    assert calls == [
        "get_claim",
        "get_policy",
        "get_claim_history",
        "list_sections",
        "read_section",
        "read_section",
        "calculate_payout",
        "record_decision",
    ]
    assert '"payout":420.0' in steps(client, "tool_result")[6]
    assert steps(client, "read") == [
        "Policy wording, 3. Trip Delay",
        "Claims manual, 9. Repeat claims",
    ]
    assert steps(client, "tool_result")[0].endswith("...")
    assert steps(client)[-1] == "Paid $420.00."
    assert len(llm.seen) == 8


def test_a_refused_section_is_not_a_read(client):
    llm = stub(
        tool_call("read_section", document="claims-manual", number="99"),
        *decided(),
    )
    work(llm)
    assert steps(client, "read") == []
    assert "read_section: Refused: The claims manual has no section 99" in steps(
        client, "tool_result"
    )


def test_a_refused_payout_comes_back_to_the_model(client):
    bad = tool_call(
        "calculate_payout",
        policy_id="POL-9999",
        benefit="baggage",
        lines=[{"description": "Bag", "amount": 50.0, "kind": "item"}],
        paid_by_others=0,
    )
    llm = stub(bad, *decided())
    work(llm)
    assert "calculate_payout: Refused: POL-9999 was not found" in steps(
        client, "tool_result"
    )
    assert isinstance(llm.seen[1][-1], ToolMessage)


def test_a_decline_ends_without_a_payment(client):
    llm = stub(DECLINE)
    assert work(llm) == "CLM-1001: decline"
    assert get(client)["paid_at"] is None
    assert saved().next == ()


def test_the_pause_before_a_payout_over_the_limit(client):
    llm = stub(decision(payout=1150.0))
    assert work(llm) == (
        "CLM-1001: approve, $1,150.00, waiting for a handler's sign-off"
    )
    held = get(client)
    assert held["status"] == "awaiting_signoff"
    assert held["paid_at"] is None
    state = saved()
    assert state.next == ("pay",)
    assert state.interrupts[0].value == {"claim_id": "CLM-1001", "payout": 1150.0}


def test_the_graph_follows_the_backend_not_its_own_limit(client):
    set_limit(2000)
    assert work(stub(decision(payout=1250.0))) == "CLM-1001: approve, $1,250.00, paid"


def test_a_signoff_resumes_the_claim_and_pays_it_once(client):
    work(stub(decision(payout=1150.0)))
    thread_id = get_thread("CLM-1001")
    signed = client.post("/api/claims/CLM-1001/signoff", json={"handler_id": "HND-01"})
    assert signed.json()["status"] == "in_progress"
    resumer = stub()
    graph.run("CLM-1001", thread_id, resumer)
    paid = get(client)
    assert paid["status"] == "decided"
    assert paid["paid_amount"] == 1150.0
    assert paid["signed_off_by"] == "HND-01"
    assert paid["signed_off_at"]
    assert resumer.seen == []
    assert saved().values["signed_off_by"] == "HND-01"
    graph.run("CLM-1001", thread_id, stub())
    assert get(client)["paid_at"] == paid["paid_at"]
    assert steps(client).count("Paid $1,150.00.") == 1


def test_a_claim_not_signed_off_keeps_waiting(client):
    work(stub(decision(payout=1150.0)))
    graph.run("CLM-1001", get_thread("CLM-1001"), stub())
    assert get(client)["status"] == "awaiting_signoff"
    assert saved().next == ("pay",)


def test_a_claim_stopped_partway_carries_on_from_its_last_step(client):
    first = stub(*READS[:2], KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        work(first)
    stopped = get(client)
    assert stopped["status"] == "in_progress"
    assert stopped["attempts"] == 0
    assert saved().next == ("agent",)

    second = stub(decision())
    line = claim.carry_on("CLM-1001", second)
    assert line == "CLM-1001: approve, $420.00, paid"
    assert len(second.seen) == 1
    carried = [m.content for m in second.seen[0] if isinstance(m, ToolMessage)]
    assert len(carried) == 2
    done = get(client)
    assert done["attempts"] == 0
    assert done["thread_id"] == stopped["thread_id"]
    assert steps(client, "attempt") == []
    calls = [s.split("(")[0] for s in steps(client, "tool_call")]
    assert calls == ["get_claim", "get_policy", "record_decision"]


def test_a_claim_being_worked_is_not_carried_on_twice(client):
    client.post("/api/claims/CLM-1001/start", json={"run_id": 1})
    llm = stub()
    with claim.working("CLM-1001"):
        assert claim.carry_on("CLM-1001", llm) is None
    assert llm.seen == []
    assert not claim.WORKING


def test_only_a_claim_in_progress_is_carried_on(client):
    assert claim.carry_on("CLM-1001", stub()) is None
    assert get(client)["status"] == "waiting"


def test_carry_on_all_picks_up_stopped_and_signed_off_claims(client):
    with pytest.raises(KeyboardInterrupt):
        work(stub(READS[0], KeyboardInterrupt()))
    work(stub(decision("CLM-1003", 3750.0)), "CLM-1003")
    client.post("/api/claims/CLM-1003/signoff", json={"handler_id": "HND-01"})
    lines = claim.carry_on_all(stub(DECLINE))
    assert lines == ["CLM-1001: decline", "CLM-1003: approve, $3,750.00, paid"]
    assert get(client)["attempts"] == 0
    assert get(client, "CLM-1003")["paid_amount"] == 3750.0


def test_a_claim_that_fails_with_an_error_starts_afresh(client):
    assert work(stub(READS[0], *DOWN)) == (
        "CLM-1001: failed (the agent stopped with an error: model unavailable), "
        "back in the queue"
    )
    failed = get(client)
    assert failed["status"] == "waiting"
    assert steps(client, "error") == [
        "The agent stopped with an error: model unavailable"
    ]
    again = stub(*decided())
    work(again)
    assert get(client)["thread_id"] != failed["thread_id"]
    assert [m.type for m in again.seen[0]] == ["system", "human"]
    assert "Attempt 2" in steps(client, "attempt")


def test_a_second_failure_leaves_the_claim_for_a_handler(client):
    work(stub(*DOWN))
    assert work(stub(*DOWN)).endswith("left for a handler")
    assert get(client)["status"] == "needs_handler"


def test_no_decision_puts_the_claim_back(client):
    assert work(stub(AIMessage(content="Decline, $0."))) == (
        "CLM-1001: failed (the agent finished without recording a decision), "
        "back in the queue"
    )
    assert get(client)["status"] == "waiting"


def test_the_call_limit_ends_a_claim(client, monkeypatch):
    monkeypatch.setattr(graph, "CALL_LIMIT", 2)
    loop = [tool_call("get_claim", claim_id="CLM-1001") for _ in range(3)]
    assert work(stub(*loop)).endswith("back in the queue")
    assert len(steps(client, "tool_call")) == 2


def test_the_call_limit_is_30():
    assert graph.CALL_LIMIT == 30


def test_a_transient_error_is_retried(client):
    llm = stub(RuntimeError("timed out"), *decided())
    assert work(llm) == "CLM-1001: approve, $100.00, paid"
    assert steps(client, "error") == []


def test_a_claim_not_waiting_is_refused(client):
    client.post("/api/claims/CLM-1001/start", json={"run_id": 1})
    llm = stub()
    assert work(llm) == "Refused: CLM-1001 is in_progress, not waiting"
    assert llm.seen == []


def test_instructions_come_from_the_backend(client, monkeypatch, tmp_path):
    (tmp_path / "claim.md").write_text("Served claim instructions.")
    (tmp_path / "queue.md").write_text("Served queue instructions.")
    monkeypatch.setattr("backend.main.INSTRUCTIONS", tmp_path)
    llm = stub(*decided())
    work(llm)
    assert llm.seen[0][0].content == "Served claim instructions."


def test_each_claim_is_traced_in_its_own_session(client, monkeypatch):
    threads = []
    monkeypatch.setattr(graph, "tracing", lambda thread: threads.append(thread) or {})
    work(stub(decision(payout=1150.0)))
    client.post("/api/claims/CLM-1001/signoff", json={"handler_id": "HND-01"})
    graph.run("CLM-1001", get_thread("CLM-1001"), stub())
    assert threads == [get_thread("CLM-1001")] * 2


def test_a_claim_is_a_trace_of_its_own_inside_a_run(client, monkeypatch):
    class Recorder(BaseCallbackHandler):
        parents: list = []

        def on_chain_start(
            self, serialized, inputs, *, run_id, parent_run_id=None, **kw
        ):
            self.parents.append(parent_run_id)

    recorder = Recorder()
    monkeypatch.setattr(graph, "tracing", lambda thread: {"callbacks": [recorder]})
    RunnableLambda(lambda _: work(stub(*decided()))).invoke(None)
    assert recorder.parents[0] is None


def test_tracing(monkeypatch):
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert common.tracing("CLM-1001-abc") == {}
    monkeypatch.setattr(common, "CallbackHandler", lambda: "handler")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    assert common.tracing("CLM-1001-abc") == {
        "callbacks": ["handler"],
        "metadata": {"langfuse_session_id": "CLM-1001-abc"},
    }


def test_checkpoints_in_memory_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert isinstance(graph.checkpointer.__wrapped__(), InMemorySaver)


def test_checkpoints_in_supabase_with_database_url(monkeypatch):
    made = {}

    class Saver:
        def __init__(self, pool):
            made["pool"] = pool

        def setup(self):
            made["setup"] = True

    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@pooler:5432/postgres")

    class Pool:
        check_connection = "check each connection"

        def __new__(cls, url, **options):
            return options

    monkeypatch.setattr(graph, "ConnectionPool", Pool)
    monkeypatch.setattr(graph, "PostgresSaver", Saver)
    graph.checkpointer.__wrapped__()
    assert made["setup"]
    assert made["pool"]["max_size"] == 4
    assert made["pool"]["check"] == "check each connection"
    assert made["pool"]["kwargs"] == {
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": graph.dict_row,
        "options": "-c search_path=undercover",
    }


def test_tools_only_call_the_backend():
    assert [t.name for t in tools.TOOLS] == [
        "get_claim",
        "get_policy",
        "get_claim_history",
        "list_sections",
        "read_section",
        "calculate_payout",
        "record_decision",
    ]
    for module in [claim, common, graph, tools]:
        source = inspect.getsource(module)
        assert "data/" not in source
        assert "open(" not in source


def test_the_backend_is_reached_by_its_ipv4_address():
    # On Windows, localhost tries IPv6 first and waits about 2 seconds a call.
    assert common.BACKEND_URL == "http://127.0.0.1:8000"


def test_server_errors_raise(monkeypatch):
    transport = httpx.MockTransport(lambda _: httpx.Response(500))
    monkeypatch.setattr(
        common,
        "backend",
        lambda: httpx.Client(transport=transport, base_url="http://x"),
    )
    with pytest.raises(httpx.HTTPStatusError):
        common.call("GET", "/api/policy-wording")


def test_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("MODEL_NAME", raising=False)
    default = common.model()
    assert default.model_name == "deepseek/deepseek-v4.1-flash"
    assert default.extra_body == {"reasoning": {"effort": "low"}}
    assert default.request_timeout == 60
    monkeypatch.setenv("MODEL_NAME", "openrouter/free")
    assert common.model().model_name == "openrouter/free"
    assert common.model().extra_body is None
