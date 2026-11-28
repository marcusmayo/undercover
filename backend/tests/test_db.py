from backend import db

SUPABASE = "postgresql://postgres.ref:secret@pooler.supabase.com:5432/postgres"


def test_sqlite_without_database_url(client, tmp_path):
    assert db.url() == f"sqlite:///{tmp_path / 'test.db'}"
    assert db.engine().dialect.name == "sqlite"


def test_postgres_with_database_url(monkeypatch):
    made = {}
    monkeypatch.setattr(
        db, "create_engine", lambda url, **options: made.update(url=url, **options)
    )
    monkeypatch.setenv("DATABASE_URL", SUPABASE)
    assert db.url() == SUPABASE
    db.make_engine(db.url())
    assert made == {
        "url": SUPABASE.replace("postgresql://", "postgresql+psycopg://"),
        "pool_size": 4,
        "max_overflow": 0,
        "pool_pre_ping": True,
        "connect_args": {"options": "-c search_path=undercover"},
    }


def test_a_real_postgres_engine_is_not_connected_until_used():
    engine = db.make_engine(SUPABASE)
    assert engine.dialect.name == "postgresql"
    assert engine.pool.size() == 4  # type: ignore[attr-defined]


def test_reseeding_puts_every_claim_back(client):
    client.post("/api/claims/CLM-1001/start", json={"run_id": 1})
    db.seed()
    assert {c["status"] for c in client.get("/api/claims").json()} == {"waiting"}
