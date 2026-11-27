"""The UnderCover database: Supabase when DATABASE_URL is set, otherwise SQLite."""

import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Column,
    Connection,
    Engine,
    Float,
    Integer,
    MetaData,
    RowMapping,
    Table,
    Text,
    create_engine,
    inspect,
    text,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEFAULT_PATH = ROOT / "backend" / "undercover.db"
SCHEMA = "undercover"
POOL_SIZE = 4

RECORD_FILES = {
    "claims": DATA / "claims.json",
    "policies": DATA / "policies.json",
    "customers": DATA / "customers.json",
    "handlers": DATA / "handlers.json",
}
SETTING_FILES = {
    "plans": DATA / "plans.json",
    "company": DATA / "company.json",
    "policy_wording": DATA / "policy_wording.md",
    "claims_manual": DATA / "claims_manual.md",
}
SEED_FILES = [*RECORD_FILES.values(), *SETTING_FILES.values()]

ID_FIELDS = {
    "claims": "claim_id",
    "policies": "policy_id",
    "customers": "customer_id",
    "handlers": "handler_id",
}

metadata = MetaData()
Table(
    "claims",
    metadata,
    Column("id", Text, primary_key=True),
    Column("record", Text, nullable=False),
    Column("status", Text, nullable=False, server_default="waiting"),
    Column("attempts", Integer, nullable=False, server_default="0"),
    Column("run_id", Integer),
    Column("thread_id", Text),
    Column("decision", Text),
    Column("payout", Float),
    Column("reason", Text),
    Column("policy_section", Text),
    Column("decided_at", Text),
    Column("signed_off_by", Text),
    Column("signed_off_at", Text),
    Column("paid_amount", Float),
    Column("paid_at", Text),
)
for name in ["policies", "customers", "handlers"]:
    Table(
        name,
        metadata,
        Column("id", Text, primary_key=True),
        Column("record", Text, nullable=False),
    )
Table(
    "settings",
    metadata,
    Column("name", Text, primary_key=True),
    Column("content", Text, nullable=False),
)
Table(
    "steps",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("claim_id", Text, nullable=False),
    Column("kind", Text, nullable=False),
    Column("content", Text, nullable=False),
    Column("created_at", Text, nullable=False),
)
Table(
    "runs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("claim_id", Text),
    Column("cap", Integer, nullable=False),
    Column("status", Text, nullable=False, server_default="running"),
    Column("todos", Text, nullable=False, server_default="[]"),
    Column("summary", Text),
    Column("problem", Text),
    Column("started_at", Text, nullable=False),
    Column("finished_at", Text),
)

engines: dict[str, Engine] = {}


def path() -> Path:
    """Return the SQLite database path, from UNDERCOVER_DB if set."""
    return Path(os.getenv("UNDERCOVER_DB", str(DEFAULT_PATH)))


def url() -> str:
    """Return the database URL: DATABASE_URL, or the SQLite file."""
    return os.getenv("DATABASE_URL") or f"sqlite:///{path()}"


def make_engine(database_url: str) -> Engine:
    """Make an engine: SQLite, or Postgres in our schema with at most 4 connections."""
    if database_url.startswith("sqlite"):
        return create_engine(database_url, connect_args={"check_same_thread": False})
    return create_engine(
        database_url.replace("postgresql://", "postgresql+psycopg://", 1),
        pool_size=POOL_SIZE,
        max_overflow=0,
        pool_pre_ping=True,
        connect_args={"options": f"-c search_path={SCHEMA}"},
    )


def engine() -> Engine:
    """Return the engine for the current database URL."""
    database_url = url()
    if database_url not in engines:
        engines[database_url] = make_engine(database_url)
    return engines[database_url]


def now() -> str:
    """Return the current time in UTC as an ISO string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def connection() -> Iterator[Connection]:
    """Yield a connection for one request, committing at the end."""
    with engine().connect() as database:
        yield database
        database.commit()


def rows(database: Connection, sql: str, **params: Any) -> list[RowMapping]:
    """Run a query and return its rows by column name."""
    return list(database.execute(text(sql), params).mappings())


def one(database: Connection, sql: str, **params: Any) -> RowMapping | None:
    """Run a query and return its first row, or None."""
    return database.execute(text(sql), params).mappings().first()


def scalar(database: Connection, sql: str, **params: Any) -> Any:
    """Run a query and return the single value it gives."""
    return database.execute(text(sql), params).scalar_one()


def run(database: Connection, sql: str, **params: Any) -> None:
    """Run a statement."""
    database.execute(text(sql), params)


def exists() -> bool:
    """Return whether the database has been seeded."""
    return inspect(engine()).has_table("claims")


def seed() -> None:
    """Rebuild our tables from the files in SEED_FILES, leaving the checkpoints."""
    with engine().begin() as database:
        if database.dialect.name == "postgresql":
            run(database, f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
        metadata.drop_all(database)
        metadata.create_all(database)
        for table, file in RECORD_FILES.items():
            records = json.loads(file.read_text(encoding="utf-8"))
            database.execute(
                text(f"INSERT INTO {table} (id, record) VALUES (:id, :record)"),  # noqa: S608
                [{"id": r[ID_FIELDS[table]], "record": json.dumps(r)} for r in records],
            )
        database.execute(
            text("INSERT INTO settings (name, content) VALUES (:name, :content)"),
            [
                {"name": name, "content": file.read_text(encoding="utf-8")}
                for name, file in SETTING_FILES.items()
            ],
        )


def setting(database: Connection, name: str) -> str:
    """Return a setting's content."""
    return scalar(
        database, "SELECT content FROM settings WHERE name = :name", name=name
    )
