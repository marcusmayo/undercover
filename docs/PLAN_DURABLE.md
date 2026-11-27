# Plan: approvals that wait, and work that survives a restart

This plan builds `docs/DURABLE.md` on the harness as it stands (`docs/STATUS.md`). The claim agent gets the payout rules as a tool and is traced in Langfuse. Our records move to Supabase. The claim agent becomes a hand-written LangGraph graph that saves its progress after every step, so it can wait for a handler's sign-off, pay once the sign-off comes, and carry on after a crash from where it stopped. The agent moves out of the backend into a small service of its own.

SIDENOTE: For consistency with the course, please read `labs/week2day3.ipynb` and build the solution in the same style as the lab. Where the lab and this brief differ, follow the brief.

The build is on the branch `week2day3-durable`, made from `week1day5-harness`.

## The approach

### The claim agent as a graph

The graph follows the lab's shape: a `ClaimState`, nodes that are plain functions returning the parts of the state they change, and conditional edges that pick the next node. `agent/graph.py` holds it.

```
START -> agent -> tools -> agent ...   (the model and its tools, as in create_agent)
                   tools -> signoff -> pay -> END   (the backend says the claim needs a sign-off)
                   tools -> pay -> END              (an approval within the limit)
                   tools -> END                     (a decline or a referral)
         agent -> END                               (no more tool calls, or 30 model calls)
```

- **State:** `messages` (with `add_messages`), `claim_id`, `model_calls`, and the outcome of the decision as the backend recorded it: `decision`, `payout`, `status`, `signed_off_by`.
- **`agent`:** calls the model with the claim instructions and the tools, adds one to `model_calls`, and posts the model's message and tool calls as steps on the claim.
- **`tools`:** runs the tool calls with LangGraph's `ToolNode` and posts each result as a step (and a `read` step for each section read). Refusals from the backend come back to the model as text, as today. When `record_decision` succeeds, the node copies the decision, payout and status the backend returned into the state.
- **`signoff`:** calls `interrupt()` with the claim id and payout. Nothing comes before the `interrupt()`, because the lab showed that a node runs again from the top when it resumes. The handler's answer (who and when) goes into the state.
- **`pay`:** calls the backend's `pay_claim` with the approved payout and posts a "Paid" step. It is its own node so it is the only thing that can run twice after a crash, and the backend makes a repeat harmless.
- **Edges:** after `agent`, go to `tools` if the model asked for one, otherwise end. After `tools`, if a decision was recorded: `signoff` when the backend's status is `awaiting_signoff`, `pay` for any other approval, and the end for a decline or a referral; otherwise back to `agent`. The graph never compares the payout with $1,000 itself: it follows the status the backend returned.

How this keeps what the middleware does today:

| Today (middleware and stream loop) | In the graph |
|---|---|
| Every step posted to the claim | `agent` and `tools` post their steps |
| Refusals returned to the model | Our tools return `Refused: ...` as text, as now |
| `ModelRetryMiddleware` (2 retries) | A LangGraph `RetryPolicy` on `agent` (3 attempts) |
| `ToolRetryMiddleware` for connection errors | A `RetryPolicy` on `tools` for `httpx.TransportError` |
| `ModelCallLimitMiddleware` | `model_calls` in the state; at 30 the edge ends the claim. The brief asks for 30, up from our 25 |

**Where this differs from the lab.** The lab has a `decide` node that asks the model for a structured decision and keeps it in the graph. We keep the `record_decision` tool, because our backend records and checks every decision, which is what makes "nothing happens twice" hold. The lab pays into a `payments` table; we pay through the backend's `pay_claim`, which already blocks a payout over the limit without a sign-off. The lab uses the claim id as the thread id; we give each attempt its own thread (below).

### The tools

The claim agent's tools become `get_claim`, `get_policy`, `get_claim_history`, `list_sections`, `read_section`, `calculate_payout` and `record_decision`. It loses `pay_claim`: paying is a step of the graph for every approval. `claim.md` changes to match: use `calculate_payout` for every approval and record the payout it returns; the payment follows on its own, after a handler's sign-off when the payout is over the limit.

### The payout rules in the backend

`labs/week2day1_payout.py` moves into `backend/payout.py`, with two changes: the plans come from the database's settings rather than the files, and the approval limit and the meal allowance come from `company.json` in the settings. A new operation runs it:

`POST /api/policies/{policy_id}/payout` (`calculate_payout`) with `{benefit, lines, paid_by_others}` returns `{calculation, payout, needs_sign_off}`.

The lab's `Line` has defaults (`quantity`, `nights`, `hours`), and our API takes no optional inputs, so every field of a line is required in the request. The agent's tool keeps the lab's `Line` with its defaults and descriptions, so the model can leave them out, and fills them in before it calls the backend. (Built: the agent's tool has its own copy of `Line`, since the agent won't import the backend once it is a service of its own.)

### Threads, attempts and Langfuse sessions

Each attempt at a claim gets its own thread. `start_claim` makes the thread id, such as `CLM-1003-1a2b3c4d`, and stores it on the claim with the run that started it. A claim stopped partway keeps its status (`in_progress`), its attempt and its thread, so it carries on from its last checkpoint. A claim that fails with an error is released as today, and its next attempt gets a new thread, so it starts afresh. A random part in the id means a reseed never meets an old checkpoint.

The entry point is `run(claim_id, thread_id)`, as `docs/TECH_STACK.md` asks. It looks at the thread's saved state and does the one right thing:

| The thread's state | What `run` does |
|---|---|
| No checkpoint yet | Starts the graph on the claim |
| Stopped partway | Carries on with `invoke(None)` from the last checkpoint |
| Waiting at `signoff`, and the claim is signed off | Resumes with `Command(resume={handler, at})`, then `pay` runs |
| Waiting at `signoff`, not signed off | Nothing; it is still waiting |

Every call passes Langfuse's `CallbackHandler` in the config with the thread id as `langfuse_session_id`, so the work before a pause and after a sign-off land in one session named after the claim and its thread. The handler is only added when the Langfuse keys are in the environment, so the tests never reach Langfuse.

### Saving progress

The agent's processes use LangGraph's synchronous `PostgresSaver` on a `psycopg_pool.ConnectionPool` of at most 4 connections, with the lab's settings (`autocommit=True`, `prepare_threshold=0`, `row_factory=dict_row`) and `search_path=undercover`, so the checkpointer's tables sit beside ours. Every `invoke` uses `durability="sync"`. Without `DATABASE_URL`, as in the tests, the graph uses LangGraph's `InMemorySaver`: the tests stop and carry on a claim within one process, so they need nothing else. (On SQLite, `npm run dev` without `DATABASE_URL` works but does not survive a restart of the agent. That is fine: the brief's durable setup is Supabase.)

### The agent's service

`agent/service.py` is a small FastAPI app, started by `npm run dev` on port 8002 beside the backend and the dashboard. It becomes the only place the agents run, so the backend no longer imports the agent (as it will be in the cloud).

| Method and path | `operation_id` | Called by |
|---|---|---|
| `GET /health` | `health` | checks |
| `POST /runs/{run_id}` | `work_run` | backend, after `start_run` or `run_agent` |
| `POST /claims/{claim_id}/resume` | `resume_claim` | backend, after `sign_off` |

Both POSTs return at once and do the work in the background. The service keeps a set of the claims it is working right now, so a claim is never worked twice at once in it.

**Catching up.** When the service starts, and at the start of every run, it carries on what was left:

1. Every claim `in_progress` that it isn't already working: a claim stopped partway, or one signed off while the agent was away. It calls `run(claim_id, thread_id)` on the claim's stored thread. A claim that then ends without a decision is released, as today.
2. When the service starts: every run still `running`. Only the service runs runs, so one left `running` was stopped. The service carries it on: a fresh queue agent on the same run, which counts the claims the run has already worked (from the `run_id` on the claims) against its cap, works the rest and finishes it.

The queue agent stays on `create_agent` and keeps its tools. `work_claim` calls `run` in place of the old `decide`. The outcome line for a held claim reads `CLM-1003: approve, $3,750.00, waiting for a handler's sign-off`.

**Stopped versus failed.** `work_claim` catches `Exception`, as today, and releases the claim, which starts afresh next time. A stop (Ctrl-C, a kill, a crash of the process) never reaches that code, so the claim stays `in_progress` with its thread and the catch-up carries it on. The tests use `KeyboardInterrupt` from the stub model to play a stop, which is exactly what Ctrl-C raises.

**`uv run -m agent`** works as now: it starts a run through the backend, which hands it to the service, and prints the to-do list as it changes. If the service isn't running, the run waits as `running` and the service picks it up when it starts.

### The backend calls the agent

The backend reads the agent's address from `UNDERCOVER_AGENT`, or `http://127.0.0.1:8002`. `begin_run` and `sign_off` commit first, then post to the service in a background task, so the handler never waits for the payment. If the service can't be reached (`httpx.TransportError`), the backend lets it go: the sign-off or the run stands, and the catch-up finds it later. That is the one place we catch an exception, because the brief asks for exactly that.

The backend's startup no longer fails running runs or releases claims in progress: those belong to the agent now, and releasing them would throw away work the agent can carry on. `UNDERCOVER_AGENT=off` goes away; the e2e tests point it at a stand-in instead (below).

### Sign-off, payment and the claim's states

- `sign_off` records who signed off and when, moves the claim to `in_progress` ("In progress" on the dashboard, as the brief asks), and tells the agent. It no longer leads to a payment from the dashboard: the dashboard's "Sign off and pay" becomes "Sign off".
- `pay_claim` moves a claim that was signed off to `decided` when it pays. The rule that blocks a payout over the limit without a sign-off stays in `payment_problem`, whatever the agent does.
- `release_claim` refuses a claim that already has a decision, so a signed-off claim whose payment hits an error stays `in_progress` for the catch-up to pay, rather than going back to the queue.

| State | How a claim gets there now |
|---|---|
| Waiting | Seeded; or its attempt failed with an error |
| In progress | `start_claim`; or a handler signed off and the payment is on its way |
| Waiting for sign-off | Approved with a payout over $1,000; the graph waits at `signoff` |
| Decided | Declined, referred, or approved and paid |
| Left for a handler | Its second attempt failed |

### Nothing happens twice

A step can run again after a restart, so the backend treats a repeat as done:

- `record_decision` on a claim that already has the same decision and payout returns the claim (200). A different decision or payout is still refused (409).
- `pay_claim` on a claim already paid the same amount returns the claim (200). A different amount is still refused.

### Our records on Supabase

`backend/db.py` moves to SQLAlchemy Core (added to the main dependencies; it is only in the `server` group today). Our tables are defined once as `Table`s on one `MetaData`, and queries are written once with `text()` and named parameters, so the same query runs on SQLite and Postgres. New runs get their id with `RETURNING`, which both support, in place of `last_insert_rowid()`.

- With `DATABASE_URL` in the environment, the engine is `postgresql+psycopg` on the session pooler, with `search_path=undercover` set on each connection, `pool_size=4` and `max_overflow=0`. Otherwise it is the SQLite file, as now.
- `npm run seed` creates the `undercover` schema if needed, drops and recreates our tables and loads `data/` (never `data/evals/`). It leaves the checkpointer's tables alone.
- The backend loads no `.env` itself. `npm run dev` and `npm run seed` run with `uv run --env-file .env`, and `agent/common.py` stops calling `load_dotenv`, so a test can never pick up `DATABASE_URL` or the Langfuse keys. The test fixtures also remove both from the environment.

Connections at most: the backend 4 and the agent's service 4, which leaves room under the brief's 12 for `npm run seed`, the Supabase MCP server and the lab.

### What `npm run dev` starts

`concurrently` with three processes: `backend` (port 8000), `agent` (`uv run --env-file .env -m uvicorn agent.service:app --port 8002`) and `dashboard` (5173). All are `uv run` or Node commands, so they work on Windows.

### The e2e tests and the stand-in

The e2e backend gets `UNDERCOVER_AGENT=http://127.0.0.1:8003`. Playwright starts `e2e/agent-standin.js` on port 8003 (built as `.js`, which is an ES module here and gets the lint config for Node files), a few lines of Node that record each POST and list them at `GET /calls`. The tests play the agent's part through the API, as they do today, and check the stand-in to see that a sign-off on the dashboard reached the agent. They never reach the service `npm run dev` starts.

### Out of scope

What the brief leaves for later: rejecting a held payout, sign-in, real payments, messages to customers and running claims in parallel. Nothing here opens `data/evals/`.

### A note on the docs

The brief calls the service "a fourth part of the agent beside the three in `docs/CODE_STANDARDS.md`". Our code standards don't list the agent's parts, so, following the brief's intent, `docs/TECH_STACK.md` will describe the agent's four parts: its tools, its graph (whose nodes take the place of the middleware), its instructions and its service.

## Phase 1: the payout rules and tracing

Move the payout rules into `backend/payout.py` and add `calculate_payout`. Give the claim agent (still `create_agent` here) the `calculate_payout` tool and update `claim.md`. `start_claim` makes the thread id and stores it with the run id. Trace each claim in Langfuse, one session per claim and thread.

- [x] `calculate_payout` for baggage, with a $239.99 suitcase and three T-shirts at $74.97 and $100 paid by others, returns the same calculation and payout as the lab's function (built on POL-2001: the lab's POL-2201 is in its own claim set, not in `data/`)
- [x] Tests cover each kind of line (item, electronics with and without Electronics Plus, cash, hotel, meals without receipts), the deductible, the benefit maximum, the trip cost cap and a payout that goes below zero
- [x] `needs_sign_off` follows `payout_approval_limit` in the settings: a test that changes the setting sees it change
- [x] A line missing a field, or an unknown benefit, returns 422; an unknown policy returns 404
- [x] `start_claim` with `{run_id}` stores the run and a new thread id like `CLM-1001-1a2b3c4d`; the claim view shows both; a retry gets a new thread id
- [x] A stub run of the claim agent calls `calculate_payout` and records its payout; `claim.md` tells the agent to do so
- [x] With Langfuse keys set (a test stubs `CallbackHandler`), the claim's config carries the handler and `langfuse_session_id` set to the thread id; without them, no handler is added
- [x] The OpenAPI test still passes: every route has an `operation_id`, a docstring and no optional input
- [x] `npm run check:python` passes with coverage of at least 80%

## Phase 2: our records on Supabase

Move `backend/db.py` and every query in `backend/main.py` to SQLAlchemy Core. Add the Postgres engine with the `undercover` schema and a pool of 4. Make `npm run seed` work on Supabase. Make repeats of a decision or a payment count as done. Run `npm run dev` and `npm run seed` with `--env-file .env`, and drop `load_dotenv` from the agent.

- [x] Every backend and agent test passes on SQLite with no network, unchanged apart from the new behavior
- [x] A test of `db` without `DATABASE_URL` uses SQLite; a test with a Postgres URL builds an engine with `search_path=undercover`, `pool_size=4` and `max_overflow=0` (checked without connecting)
- [x] `record_decision` repeated with the same decision and payout returns 200 and the claim, with one decision recorded; a different decision returns 409
- [x] `pay_claim` repeated with the same amount returns 200 and the claim, paid once; a different amount returns 409
- [x] `npm run seed` with `.env` in place reseeds Supabase: through the Supabase MCP server, `undercover.claims` has 25 rows, all `waiting`, and `undercover` holds our tables; running it again gives the same result
- [x] `npm run dev`: the dashboard on http://localhost:5173 shows the 25 claims from Supabase (checked with the Playwright MCP server)
- [x] `npm run check` passes with `.env` moved out of the way

## Phase 3: the claim agent as a graph

Build `agent/graph.py`: `ClaimState`, the nodes `agent`, `tools`, `signoff` and `pay`, the edges, the retry policies, the 30-call cap and the checkpointer (`PostgresSaver` on a pool of 4 with `DATABASE_URL`, `InMemorySaver` without). Build the entry point `run(claim_id, thread_id)`. Remove `pay_claim` from the tools and update `claim.md`. Change the backend: `sign_off` records the sign-off and moves the claim to `in_progress`; `pay_claim` moves a signed-off claim to `decided`; `release_claim` refuses a claim with a decision.

- [x] The claim agent's tools are exactly the seven listed above; `pay_claim` is not among them
- [x] An approval within the limit (stub): the graph records the decision, then `pay` pays it; the claim is `decided` and paid, with a "Paid" step
- [x] A decline (stub): the graph ends after the decision and nothing is paid
- [x] The pause: an approval over $1,000 (stub) leaves the claim `awaiting_signoff`, unpaid, with the thread waiting at `signoff`; `work_claim` returns `..., waiting for a handler's sign-off`
- [x] The graph follows the backend: a test that raises the limit in the settings sees the same $1,250 approval go straight to `pay`
- [x] The resume: after `sign_off`, the claim is `in_progress` with the handler and time; `run` on its thread resumes at `signoff`, pays once and the claim is `decided` with `paid_at`, `signed_off_by` and `signed_off_at` set; a second `run` changes nothing
- [x] Stopped partway: a stub that raises `KeyboardInterrupt` after the claim's first tool calls leaves the claim `in_progress`; `run` on the same thread carries on from the last checkpoint without repeating the model calls before it, with the attempts unchanged and one decision
- [x] A failure: a stub that raises an error partway releases the claim to `waiting`; its next attempt has a new thread and starts from the beginning
- [x] 30 model calls without a decision end the claim, and it is released
- [x] A transient model error (fails once, then succeeds) is retried by the retry policy and the claim is decided
- [x] Steps: a stub run posts the same kinds of step as before (`tool_call`, `tool_result`, `message`, `read`), and refusals come back to the model as text
- [x] `pay_claim` over the limit without a sign-off is still refused, whatever calls it
- [x] `release_claim` on a claim with a decision returns 409
- [x] The queue agent's tests still pass with `work_claim` on the graph
- [x] `npm run check:python` passes

## Phase 4: the agent's service

Build `agent/service.py` with `health`, `work_run` and `resume_claim`, the set of claims being worked, and the catch-up on startup and at the start of each run. Carry on a stopped run with a fresh queue agent that counts the run's claims already worked. The backend posts to the service after `start_run`, `run_agent` and `sign_off`, and no longer runs the agent or releases claims on startup. Add the agent to `npm run dev`.

- [x] `resume_claim` (stub model, the backend in-process) on a signed-off claim pays it once; a second call does nothing
- [x] `work_run` works the run in the background and returns at once
- [x] The catch-up: with one claim stopped partway and one signed off while the service was down, starting the service carries both on: the first is decided with the same thread and attempts, the second is paid once
- [x] A stopped run is finished: a run left `running` with one of its two claims worked is carried on when the service starts; it works one more claim, not two, and finishes with a summary
- [x] A run starts with the catch-up: a signed-off claim waiting is paid before the run's first claim
- [x] The backend's `sign_off` commits, then posts `resume_claim` to `UNDERCOVER_AGENT`; with the service unreachable, the sign-off still returns 200 and stands (backend tests use a recording fake for the call)
- [x] `start_run` and `run_agent` post `work_run`; the backend no longer imports `agent`
- [x] Restarting the backend leaves a claim `in_progress` as it is
- [x] `npm run dev` starts three processes; `GET http://127.0.0.1:8002/health` returns `{"status": "ok"}`
- [x] `uv run -m agent --cap 1` with `MODEL_NAME` unset starts a run that the service works and prints its summary (a live call; reseed first)
- [x] `npm run check:python` passes

## Phase 5: the dashboard and the e2e tests

The sign-off button becomes "Sign off" and no longer pays. A claim signed off and not yet paid shows as In progress, with "Signed off by Dana Whitfield at ..., payment on its way". A paid claim shows the payment with the handler and the time of the sign-off. Add the stand-in on port 8003 and point the e2e backend at it.

- [x] Vitest: signing off sends one request, the sign-off, and never a payment; a signed-off unpaid claim shows the handler, the time and that the payment is on its way; a paid claim shows the payment with the handler and the time
- [x] e2e: a claim played to `awaiting_signoff` through the API is signed off on the dashboard; the stand-in's `/calls` lists `POST /claims/<id>/resume`; the claim shows In progress
- [x] e2e: the payment played through the API then shows on the claim page as paid, with the handler and the time
- [x] e2e: starting a run from the queue page reaches the stand-in as `POST /runs/<id>`
- [x] The earlier e2e tests still pass, on the stand-in in place of `UNDERCOVER_AGENT=off`
- [x] Checked with the Playwright MCP server in light and dark mode: the sign-off panel and the payment read well and the brand holds
- [x] `npm run check` passes with `.env` moved out of the way

## Phase 6: the docs and the live checks on Supabase

Update `docs/SPEC.md` (its section on working the queue, not its Rules), `docs/TECH_STACK.md`, `docs/PROCESS.md` and `docs/STATUS.md`. Then reseed Supabase and run the real model, capped at five claims.

- [ ] `docs/SPEC.md`, `docs/TECH_STACK.md`, `docs/PROCESS.md` and `docs/STATUS.md` describe the product as it now is; the Rules in `docs/SPEC.md` are unchanged
- [ ] Live run: `npm run seed`, `npm run dev`, then Run the queue with a cap of 5 on the default model works five claims and the other 20 stay `waiting`
- [ ] If none of the five needs a sign-off, a one-claim run on a claim over the limit (such as CLM-1003) gives one, after a reseed
- [ ] The held claim waits as Waiting for sign-off. Signed off on the dashboard with the Playwright MCP server, it shows as paid within five seconds, with the handler and the time
- [ ] During the run, through the Supabase MCP server, `SELECT count(*) FROM pg_stat_activity WHERE usename = 'postgres'` stays at 12 or fewer
- [ ] Through Langfuse's API, the held claim's session (named for its thread) has traces from before and after the sign-off
- [ ] The kill: after a reseed, a one-claim run is stopped by killing the agent's service after the claim's first steps; when the service starts again it carries on the same thread, with the attempts unchanged and one decision, and the run finishes
- [ ] `npm run check` passes in full with `.env` moved out of the way
