import pytest
from fastapi.testclient import TestClient

from backend import db
from backend.main import app


@pytest.fixture
def told(monkeypatch):
    """What the backend told the agent's service, in place of telling it."""
    calls = []
    monkeypatch.setattr("backend.main.tell_agent", calls.append)
    return calls


@pytest.fixture
def client(tmp_path, monkeypatch, told):
    for name in ["DATABASE_URL", "LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("UNDERCOVER_DB", str(tmp_path / "test.db"))
    db.seed()
    return TestClient(app)
