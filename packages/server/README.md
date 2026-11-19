# @supertuner/server

Control-plane HTTP server for SuperTuner. It owns the run queue, SQLite state,
JSONL event persistence and the SSE stream; the Python engine in `engine/` owns
all memory math and training. The API contract is ARCHITECTURE.md §9.

```
src/
  index.ts      buildServer() + main() and graceful shutdown
  config.ts     SUPERTUNER_* environment -> ServerConfig, argv parsing, path roots
  db.ts         node:sqlite RunStore, migrations, interrupted-run recovery
  engine.ts     engine CLI client (compile/plugins/env/families/version, argv only)
  supervisor.ts queue, one child per run, events.jsonl, SSE fan-out, cancel
  routes.ts     every /api route, SSE, artifacts, static web dist
  artifacts.ts  output/ scanning
  process.ts    SIGTERM/SIGKILL and Windows taskkill /T /F helpers
```

## Requirements

- Node >= 22.5 for `node:sqlite`. Node 22.5–23.x may still require
  `node --experimental-sqlite`; Node 26 ships it unflagged. On a runtime without
  it, startup fails with `SUPERTUNER DB UNAVAILABLE: this Node runtime has no
  'node:sqlite' ...` instead of a module-resolution stack.
- The engine reachable through `SUPERTUNER_ENGINE_CMD` for any route that
  probes it. Without one, `/api/health` and `/api/runs*` still work but
  `/api/system`, `/api/plugins`, `/api/families`, `/api/plan` and
  `POST /api/runs` return 503.

## Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `SUPERTUNER_HOME` | `~/.supertuner` | state root: `runs/`, `tmp/`, database |
| `SUPERTUNER_ENGINE_CMD` | `uv run --project <repo>/engine supertuner-engine` | engine argv (quoted path allowed; no shell) |
| `SUPERTUNER_HOST` | `127.0.0.1` | bind address |
| `SUPERTUNER_PORT` | `8787` | bind port |
| `SUPERTUNER_WEB_DIST` | `<repo>/packages/web/dist` | static UI; enabled only if `index.html` exists |
| `SUPERTUNER_MAX_CONCURRENT_RUNS` | `1` | queue width |
| `SUPERTUNER_ALLOW_ANY_PATH` | `0` | allow dataset/output paths outside the roots |
| `SUPERTUNER_ALLOWED_ROOTS` | `home`, datasets, models, runs, artifacts | extra roots, `path.delimiter`-separated |
| `SUPERTUNER_DB_PATH` | `<home>/supertuner.db` | SQLite file |
| `SUPERTUNER_MODELS_DIR` / `SUPERTUNER_DATASETS_DIR` | `<home>/models`, `<home>/datasets` | reported paths |
| `SUPERTUNER_COMPILE_TIMEOUT_MS` / `SUPERTUNER_PROBE_TIMEOUT_MS` | `120000` / `60000` | engine deadlines |
| `SUPERTUNER_CANCEL_GRACE_MS` | `10000` | SIGTERM → taskkill/SIGKILL grace |
| `SUPERTUNER_LOG_LEVEL` | `info` | pino level |

Paths in a run spec must be absolute and, unless `SUPERTUNER_ALLOW_ANY_PATH=1`,
inside one of the allowed roots. `~` is expanded; relative paths are rejected
(`422` with `code: "path.relative"` / `"path.outside_roots"`).

## Routes

| Method | Path | Result |
| --- | --- | --- |
| GET | `/api/health` | `{ok, version}` |
| GET | `/api/system` | env probe, engine/server versions, paths |
| GET | `/api/families` | `{llm, video}` model families |
| GET | `/api/plugins` | `{plugins, errors}` |
| GET | `/api/recipes` | `{recipes}` from `@supertuner/core` |
| POST | `/api/plan` | compiles `{spec}`, returns the `RunPlan` |
| POST | `/api/runs` | re-compiles, persists, queues; `201` or `422 {error, issues, plan}` |
| GET | `/api/runs?limit=` | `{runs}` newest first |
| GET | `/api/runs/:id` | `RunRecord` |
| GET | `/api/runs/:id/events` | JSON replay of `events.jsonl` |
| GET | `/api/runs/:id/events/stream` | SSE: replay then live, heartbeat every 15 s |
| POST | `/api/runs/:id/cancel` | queued → dropped, running → `<run>/cancel` + SIGTERM, then hard kill |
| GET | `/api/runs/:id/artifacts` | files under `<run>/output/` |
| POST | `/api/engine/reload` | clears probe caches, returns plugins |

Error shape: `{error: string, issues?: Issue[]}`. Engine failures map to
`422` (refused spec), `503` (engine missing/unsupported command), `502`
(protocol/crash) and `504` (timeout). CORS is enabled for
`http://localhost:5173` and `http://127.0.0.1:5173` only.

## Run layout

```
<home>/runs/<id>/
  spec.json        normalized spec the engine re-compiles
  plan.json        server-side compile result
  events.jsonl     one validated RunEvent per line (server is the only writer)
  cancel           created on cancel; the engine polls it
  logs/engine.stderr.log
  logs/engine.stdout.log  malformed stdout lines ("[protocol] ...")
  output/          artifacts advertised by the run
```

At startup every row left `queued`/`running` is marked `failed` with
`server restarted`, because its child process cannot survive a restart.

## Development

```sh
pnpm --filter @supertuner/core build      # server types come from core's dist
pnpm --filter @supertuner/server typecheck
pnpm --filter @supertuner/server test
pnpm --filter @supertuner/server build
pnpm --filter @supertuner/server dev      # tsx watch
```

Tests use a temp `SUPERTUNER_HOME` and `test/fixtures/fake-engine.mjs`, never
the real home. SSE and cancel tests listen on port `0` (dynamic).

## Known gaps

- `families --json` and `plugins --json` depend on engine builds that
  implement them; until then those routes return `503` with the engine's usage
  message.
- Future engine event types are persisted and streamed verbatim (parse
  fallback) so `phase_start`/`phase_end` work before core knows them.