# Tech stack

- `backend/`: FastAPI with SQLAlchemy Core. Our records are on Supabase's Postgres when `DATABASE_URL` is set (the session pooler, in the schema `undercover`, at most 4 connections), and in SQLite otherwise, as in the tests. Each query is written once for both. Every API operation has an `operation_id`, a one-line docstring and no optional inputs. Paying a claim is a separate operation that takes the amount. The payout rules are in `backend/payout.py`, behind `calculate_payout`, which takes the approval limit from `data/company.json`. The backend tells the agent's service about each new run and each sign-off.
- `agent/`: the UnderCover Agent on OpenRouter, in four parts:
  - its tools (`agent/tools.py`), which reach claims and the instructions only through the backend's API
  - the claim agent, a hand-written LangGraph graph (`agent/graph.py`) whose nodes take the place of middleware: it posts every step to the claim, returns refusals to the model, retries model errors, caps a claim at 30 model calls, pays every approval, and waits at `interrupt()` for a sign-off when the backend says a claim needs one. Its entry point is `run(claim_id, thread_id)`. LangGraph's synchronous Postgres checkpointer saves its progress after every step (`durability="sync"`) in the same Supabase database, or in memory without `DATABASE_URL`. The queue agent, which plans a run as a to-do list and hands each claim to the graph, is LangChain's `create_agent`.
  - its instructions (`backend/instructions/`), served by the backend
  - its service (`agent/service.py`), a small FastAPI app on `UNDERCOVER_AGENT` or `http://127.0.0.1:8002`, which works runs, picks claims back up after a sign-off, and carries on stopped work when it starts
  It uses `deepseek/deepseek-v4.1-flash` with low reasoning effort by default, or the model in the environment variable `MODEL_NAME` if set, such as `openrouter/free`. Each claim's work is traced in Langfuse, one session per claim and thread, when the Langfuse keys are set.
- `dashboard/`: React, Vite and TypeScript.
- `e2e/`: Playwright tests, with a stand-in for the agent's service on port 8003.
- `data/`: the business data the backend starts from. The agent never sees `data/evals/`, which holds the answer key.

Python 3.12 with uv, all in the root `pyproject.toml`. Node 26. The keys are in `.env`: `OPENROUTER_API_KEY`, `DATABASE_URL` and `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_BASE_URL`.
