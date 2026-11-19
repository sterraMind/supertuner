/**
 * Server entry point.
 *
 * `buildServer` is the testable composition root: it owns the database, the
 * engine client and the supervisor, and returns them so tests can inspect
 * state without HTTP. `main` only adds listening and signal handling.
 */

import { mkdirSync } from "node:fs";
import { pathToFileURL } from "node:url";
import Fastify, { type FastifyInstance } from "fastify";
import { loadServerConfig, type ServerConfig } from "./config.js";
import { RunStore } from "./db.js";
import { EngineClient } from "./engine.js";
import { registerRoutes } from "./routes.js";
import { Supervisor, type LogSink } from "./supervisor.js";

export const SERVER_VERSION = "0.1.0";

export interface BuildServerOptions {
  /** Pre-built config; otherwise derived from `env`. */
  config?: ServerConfig;
  env?: NodeJS.ProcessEnv;
  /** `true` for pino logging at the configured level, or a custom sink. */
  logger?: boolean | LogSink;
}

export interface BuiltServer {
  app: FastifyInstance;
  config: ServerConfig;
  store: RunStore;
  engine: EngineClient;
  supervisor: Supervisor;
  close: () => Promise<void>;
}

export async function buildServer(options: BuildServerOptions = {}): Promise<BuiltServer> {
  const config = options.config ?? loadServerConfig({ env: options.env });
  mkdirSync(config.home, { recursive: true });
  mkdirSync(config.runsDir, { recursive: true });

  const store = new RunStore(config.dbPath);
  // Rows left queued/running belong to a process that no longer exists.
  store.recoverInterrupted("server restarted");

  const app = Fastify({
    logger: options.logger === true ? { level: config.logLevel } : false,
    bodyLimit: 8 * 1024 * 1024,
    trustProxy: false,
  });

  const sink: LogSink =
    options.logger !== undefined && typeof options.logger === "object"
      ? options.logger
      : {
          debug: (message) => app.log.debug(message),
          info: (message) => app.log.info(message),
          warn: (message) => app.log.warn(message),
          error: (message) => app.log.error(message),
        };

  const engine = new EngineClient({ config });
  const supervisor = new Supervisor({ config, engine, store, logger: sink });
  await registerRoutes(app, { config, store, engine, supervisor, serverVersion: SERVER_VERSION });

  let closing = false;
  const close = async (): Promise<void> => {
    if (closing) return;
    closing = true;
    try {
      await app.close();
    } finally {
      try {
        await supervisor.shutdown();
      } finally {
        store.close();
      }
    }
  };

  return { app, config, store, engine, supervisor, close };
}

async function main(): Promise<void> {
  const built = await buildServer({ logger: true });
  const { host, port } = built.config;
  await built.app.listen({ host, port });
  built.app.log.info(`SuperTuner server listening on http://${host}:${port} (home ${built.config.home})`);
  if (built.config.webDist === null) {
    built.app.log.warn("packages/web/dist not found; API-only mode (run `pnpm --filter @supertuner/web build`)");
  }

  let stopping = false;
  const stop = async (signal: string): Promise<void> => {
    if (stopping) return;
    stopping = true;
    built.app.log.info(`${signal} received; cancelling runs and closing`);
    try {
      await built.close();
    } finally {
      process.exit(0);
    }
  };
  process.on("SIGINT", () => void stop("SIGINT"));
  process.on("SIGTERM", () => void stop("SIGTERM"));
}

const entry = process.argv[1];
if (entry !== undefined && import.meta.url === pathToFileURL(entry).href) {
  main().catch((cause: unknown) => {
    console.error(cause instanceof Error ? cause.stack ?? cause.message : cause);
    process.exitCode = 1;
  });
}