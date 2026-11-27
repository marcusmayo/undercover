import pytest
from fastapi.testclient import TestClient

from backend import db
from backend.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    for name in ["DATABASE_URL", "LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("UNDERCOVER_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("UNDERCOVER_AGENT", "off")
    db.seed()
    return TestClient(app)
