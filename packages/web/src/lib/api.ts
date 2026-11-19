/**
 * Typed client for the SuperTuner HTTP API (ARCHITECTURE §9).
 *
 * Every response is validated with the schemas that live in `@supertuner/core`,
 * so the UI never trusts the wire format. Failures surface as `ApiError`, which
 * carries the server's `issues` array so pages can point at the offending field.
 *
 * The API host is never hardcoded: `VITE_API_BASE` (server origin) is used when
 * present, otherwise the dev proxy / same-origin `/api` path.
 */

import {
  FamiliesResponseSchema,
  PluginsResponseSchema,
  RecipeSchema,
  RunPlanSchema,
  RunRecordSchema,
  SystemResponseSchema,
  type FamiliesResponse,
  type Issue,
  type Artifact,
  type PluginManifest,
  type PluginsResponse,
  type Recipe,
  type RunPlan,
  type RunRecord,
  type RunSpec,
  type SystemResponse,
} from "@supertuner/core";
import { RunEventSchema, parseEventLine, type RunEvent } from "@supertuner/core/events";

export interface HealthResponse {
  ok: boolean;
  version: string;
}

export interface ApiErrorInit {
  status: number;
  issues?: Issue[];
  payload?: unknown;
  url?: string;
}

/** An HTTP or contract failure. `issues` is empty when the server sent none. */
export class ApiError extends Error {
  readonly status: number;
  readonly issues: Issue[];
  readonly payload: unknown;
  readonly url: string;

  constructor(message: string, init: ApiErrorInit) {
    super(message);
    this.name = "ApiError";
    this.status = init.status;
    this.issues = init.issues ?? [];
    this.payload = init.payload;
    this.url = init.url ?? "";
  }

  /** Single-line summary for toasts and inline error text. */
  get summary(): string {
    if (this.issues.length === 0) return this.message;
    return `${this.message} (${this.issues.length} issue${this.issues.length === 1 ? "" : "s"})`;
  }
}

/** Human-readable text for anything that was thrown. */
export function formatApiError(error: unknown): string {
  if (error instanceof ApiError) {
    const first = error.issues[0];
    if (first) return `${error.message}: ${first.message}`;
    return error.message;
  }
  if (error instanceof Error) return error.message;
  return String(error);
}

function issueFromUnknown(value: unknown): Issue | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  const severity = raw.severity;
  if (severity !== "error" && severity !== "warning" && severity !== "info") return null;
  return {
    severity,
    code: typeof raw.code === "string" ? raw.code : "issue",
    message: typeof raw.message === "string" ? raw.message : String(raw.message ?? "Unknown issue"),
    hint: typeof raw.hint === "string" ? raw.hint : "",
    field: typeof raw.field === "string" ? raw.field : null,
  };
}

function issuesFromPayload(payload: unknown): Issue[] {
  if (!payload || typeof payload !== "object") return [];
  const raw = payload as Record<string, unknown>;
  if (!Array.isArray(raw.issues)) return [];
  const issues: Issue[] = [];
  for (const entry of raw.issues) {
    const issue = issueFromUnknown(entry);
    if (issue) issues.push(issue);
  }
  return issues;
}

/** zod-shaped issues -> contract issues, so a contract drift is legible too. */
function issuesFromValidation(error: unknown): Issue[] {
  if (!error || typeof error !== "object") return [];
  const raw = error as { issues?: unknown };
  if (!Array.isArray(raw.issues)) return [];
  const issues: Issue[] = [];
  for (const entry of raw.issues) {
    if (!entry || typeof entry !== "object") continue;
    const item = entry as { path?: unknown; message?: unknown; code?: unknown };
    const path = Array.isArray(item.path) ? item.path.join(".") : "";
    issues.push({
      severity: "error",
      code: typeof item.code === "string" ? item.code : "invalid_response",
      message: typeof item.message === "string" ? item.message : "Response did not match the contract",
      hint: path ? `field: ${path}` : "",
      field: path || null,
    });
  }
  return issues;
}

export function apiOrigin(baseUrl?: string): string {
  const fromEnv = baseUrl ?? (import.meta.env?.VITE_API_BASE as string | undefined);
  const trimmed = (fromEnv ?? "").trim().replace(/\/+$/, "");
  return trimmed;
}

/** Absolute (or root-relative) URL for an API path such as "/api/health". */
export function apiUrl(path: string, baseUrl?: string): string {
  const origin = apiOrigin(baseUrl);
  const suffix = path.startsWith("/") ? path : `/${path}`;
  const root = origin.endsWith("/api") ? origin : `${origin}/api`;
  return `${root}${suffix.replace(/^\/api/, "")}`;
}

export interface ApiClientOptions {
  /** Server origin, e.g. "http://127.0.0.1:8787". Defaults to VITE_API_BASE or same-origin. */
  baseUrl?: string;
  fetchImpl?: typeof fetch;
  EventSourceImpl?: EventSourceConstructor;
}

export type EventSourceConstructor = new (url: string) => EventSource;

export interface ApiClient {
  readonly baseUrl: string;
  health(signal?: AbortSignal): Promise<HealthResponse>;
  system(signal?: AbortSignal): Promise<SystemResponse>;
  families(signal?: AbortSignal): Promise<FamiliesResponse>;
  plugins(signal?: AbortSignal): Promise<PluginsResponse>;
  recipes(signal?: AbortSignal): Promise<Recipe[]>;
  plan(spec: RunSpec, signal?: AbortSignal): Promise<RunPlan>;
  createRun(spec: RunSpec, signal?: AbortSignal): Promise<RunRecord>;
  listRuns(limit?: number, signal?: AbortSignal): Promise<RunRecord[]>;
  getRun(id: string, signal?: AbortSignal): Promise<RunRecord>;
  getEvents(id: string, signal?: AbortSignal): Promise<RunEvent[]>;
  getArtifacts(id: string, signal?: AbortSignal): Promise<Artifact[]>;
  cancelRun(id: string, signal?: AbortSignal): Promise<void>;
  reloadEngine(signal?: AbortSignal): Promise<unknown>;
  openRunStream(id: string, handlers: StreamHandlers, options?: StreamOptions): RunStream;
}

interface SchemaLike<T> {
  parse(data: unknown): T;
}

export function createApiClient(options: ApiClientOptions = {}): ApiClient {
  const origin = apiOrigin(options.baseUrl);
  const doFetch: typeof fetch =
    options.fetchImpl ?? ((input, init) => fetch(input, init));

  async function request<T>(
    path: string,
    schema: SchemaLike<T>,
    init?: RequestInit & { signal?: AbortSignal },
  ): Promise<T> {
    const url = apiUrl(path, origin);
    let response: Response;
    try {
      response = await doFetch(url, {
        ...init,
        headers: {
          Accept: "application/json",
          ...(init?.body ? { "Content-Type": "application/json" } : {}),
          ...(init?.headers ?? {}),
        },
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") throw error;
      throw new ApiError(error instanceof Error ? error.message : "Network request failed", {
        status: 0,
        url,
      });
    }

    const text = await response.text();
    let payload: unknown = null;
    if (text.length > 0) {
      try {
        payload = JSON.parse(text) as unknown;
      } catch {
        payload = text;
      }
    }

    if (!response.ok) {
      const serverMessage =
        payload && typeof payload === "object" && typeof (payload as Record<string, unknown>).error === "string"
          ? ((payload as Record<string, unknown>).error as string)
          : `Request failed with status ${response.status}`;
      throw new ApiError(serverMessage, {
        status: response.status,
        issues: issuesFromPayload(payload),
        payload,
        url,
      });
    }

    try {
      return schema.parse(payload);
    } catch (error) {
      throw new ApiError("The server response did not match the expected schema", {
        status: response.status,
        issues: issuesFromValidation(error),
        payload,
        url,
      });
    }
  }

  async function jsonRequest<T>(path: string, init?: RequestInit & { signal?: AbortSignal }): Promise<T> {
    const url = apiUrl(path, origin);
    let response: Response;
    try {
      response = await doFetch(url, {
        ...init,
        headers: {
          Accept: "application/json",
          ...(init?.body ? { "Content-Type": "application/json" } : {}),
          ...(init?.headers ?? {}),
        },
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") throw error;
      throw new ApiError(error instanceof Error ? error.message : "Network request failed", {
        status: 0,
        url,
      });
    }
    const text = await response.text();
    let payload: unknown = null;
    if (text.length > 0) {
      try {
        payload = JSON.parse(text) as unknown;
      } catch {
        payload = text;
      }
    }
    if (!response.ok) {
      const serverMessage =
        payload && typeof payload === "object" && typeof (payload as Record<string, unknown>).error === "string"
          ? ((payload as Record<string, unknown>).error as string)
          : `Request failed with status ${response.status}`;
      throw new ApiError(serverMessage, {
        status: response.status,
        issues: issuesFromPayload(payload),
        payload,
        url,
      });
    }
    return payload as T;
  }

  /** Validates a jsonRequest payload, converting zod failures into ApiError. */
  function parsePayload<T>(schema: SchemaLike<T>, payload: unknown, path: string): T {
    try {
      return schema.parse(payload);
    } catch (error) {
      throw new ApiError("The server response did not match the expected schema", {
        status: 200,
        issues: issuesFromValidation(error),
        payload,
        url: apiUrl(path, origin),
      });
    }
  }

  const client: ApiClient = {
    baseUrl: origin,

    async health(signal) {
      const payload = await jsonRequest<unknown>("/api/health", { signal });
      const raw = (payload ?? {}) as Record<string, unknown>;
      return {
        ok: raw.ok === true,
        version: typeof raw.version === "string" ? raw.version : "",
      };
    },

    system: (signal) => request("/api/system", SystemResponseSchema, { signal }),

    families: (signal) => request("/api/families", FamiliesResponseSchema, { signal }),

    plugins: (signal) => request("/api/plugins", PluginsResponseSchema, { signal }),

    async recipes(signal) {
      const payload = await jsonRequest<unknown>("/api/recipes", { signal });
      const list = Array.isArray(payload)
        ? payload
        : payload && typeof payload === "object" && Array.isArray((payload as Record<string, unknown>).recipes)
          ? ((payload as Record<string, unknown>).recipes as unknown[])
          : [];
      return parsePayload(RecipeSchema.array(), list, "/api/recipes");
    },

    plan: (spec, signal) =>
      request(
        "/api/plan",
        RunPlanSchema,
        { method: "POST", body: JSON.stringify({ spec }), signal },
      ),

    createRun: (spec, signal) =>
      request(
        "/api/runs",
        RunRecordSchema,
        { method: "POST", body: JSON.stringify({ spec }), signal },
      ),

    async listRuns(limit, signal) {
      const query = limit === undefined ? "" : `?limit=${encodeURIComponent(String(limit))}`;
      const payload = await jsonRequest<unknown>(`/api/runs${query}`, { signal });
      const list = Array.isArray(payload)
        ? payload
        : payload && typeof payload === "object" && Array.isArray((payload as Record<string, unknown>).runs)
          ? ((payload as Record<string, unknown>).runs as unknown[])
          : [];
      return parsePayload(RunRecordSchema.array(), list, "/api/runs");
    },

    getRun: (id, signal) => request(`/api/runs/${encodeURIComponent(id)}`, RunRecordSchema, { signal }),

    async getEvents(id, signal) {
      const payload = await jsonRequest<unknown>(`/api/runs/${encodeURIComponent(id)}/events`, { signal });
      const list = Array.isArray(payload)
        ? payload
        : payload && typeof payload === "object" && Array.isArray((payload as Record<string, unknown>).events)
          ? ((payload as Record<string, unknown>).events as unknown[])
          : [];
      const events: RunEvent[] = [];
      for (const entry of list) {
        const parsed = RunEventSchema.safeParse(entry);
        if (parsed.success) events.push(parsed.data);
      }
      return events;
    },

    async getArtifacts(id, signal) {
      const payload = await jsonRequest<unknown>(`/api/runs/${encodeURIComponent(id)}/artifacts`, { signal });
      const list = Array.isArray(payload)
        ? payload
        : payload && typeof payload === "object" && Array.isArray((payload as Record<string, unknown>).artifacts)
          ? ((payload as Record<string, unknown>).artifacts as unknown[])
          : [];
      return parseArtifacts(list);
    },

    async cancelRun(id, signal) {
      await jsonRequest<unknown>(`/api/runs/${encodeURIComponent(id)}/cancel`, {
        method: "POST",
        signal,
      });
    },

    reloadEngine: (signal) => jsonRequest<unknown>("/api/engine/reload", { method: "POST", signal }),

    openRunStream: (id, handlers, streamOptions) =>
      openRunStream(id, handlers, { baseUrl: origin, EventSourceImpl: options.EventSourceImpl, ...streamOptions }),
  };

  return client;
}

/* ---------------------------------------------------------------------------
 * Server-sent events
 * ------------------------------------------------------------------------- */

export type StreamStatus = "connecting" | "open" | "reconnecting" | "closed" | "failed";

export interface StreamHandlers {
  onEvent(event: RunEvent): void;
  onOpen?(): void;
  onStatus?(status: StreamStatus, detail?: { attempt: number; retryInMs: number | null }): void;
  onError?(message: string): void;
}

export interface StreamOptions {
  baseUrl?: string;
  EventSourceImpl?: EventSourceConstructor;
  /** Reconnect attempts before giving up. */
  maxRetries?: number;
  /** Backoff ladder in milliseconds; the last entry repeats. */
  backoffMs?: number[];
  setTimeoutImpl?: (handler: () => void, timeout: number) => number;
  clearTimeoutImpl?: (handle: number) => void;
}

export interface RunStream {
  close(): void;
  readonly closed: boolean;
}

const DEFAULT_BACKOFF = [1000, 2000, 4000, 8000, 15000];

/**
 * Opens `GET /api/runs/:id/events/stream` and feeds parsed `RunEvent`s to the
 * caller. Reconnects with exponential backoff when the socket drops, and stops
 * for good once the run reports `run_end`.
 */
export function openRunStream(
  runId: string,
  handlers: StreamHandlers,
  options: StreamOptions = {},
): RunStream {
  const Ctor = options.EventSourceImpl ?? (globalThis.EventSource as EventSourceConstructor | undefined);
  const maxRetries = options.maxRetries ?? 8;
  const backoff = options.backoffMs ?? DEFAULT_BACKOFF;
  const schedule = options.setTimeoutImpl ?? ((fn: () => void, ms: number) => setTimeout(fn, ms) as unknown as number);
  const unschedule = options.clearTimeoutImpl ?? ((handle: number) => clearTimeout(handle));

  let source: EventSource | null = null;
  let timer: number | null = null;
  let attempt = 0;
  let closed = false;

  const url = apiUrl(`/api/runs/${encodeURIComponent(runId)}/events/stream`, options.baseUrl);

  const cleanup = (): void => {
    if (timer !== null) {
      unschedule(timer);
      timer = null;
    }
    if (source) {
      source.close();
      source = null;
    }
  };

  const connect = (): void => {
    if (closed) return;
    if (!Ctor) {
      handlers.onStatus?.("failed", { attempt, retryInMs: null });
      handlers.onError?.("This browser has no EventSource support");
      return;
    }
    handlers.onStatus?.(attempt === 0 ? "connecting" : "reconnecting", { attempt, retryInMs: null });
    const es = new Ctor(url);
    source = es;

    es.onopen = () => {
      attempt = 0;
      handlers.onStatus?.("open", { attempt: 0, retryInMs: null });
      handlers.onOpen?.();
    };

    es.onmessage = (message: MessageEvent) => {
      const parsed = parseEventLine(typeof message.data === "string" ? message.data : String(message.data));
      if (!parsed.ok) {
        handlers.onError?.(parsed.reason);
        return;
      }
      handlers.onEvent(parsed.event);
      if (parsed.event.type === "run_end") {
        closed = true;
        cleanup();
        handlers.onStatus?.("closed", { attempt: 0, retryInMs: null });
      }
    };

    es.onerror = () => {
      if (closed) return;
      es.close();
      if (source === es) source = null;
      if (attempt >= maxRetries) {
        closed = true;
        handlers.onStatus?.("failed", { attempt, retryInMs: null });
        handlers.onError?.("Lost the event stream and could not reconnect");
        return;
      }
      const delay = backoff[Math.min(attempt, backoff.length - 1)] ?? 1000;
      attempt += 1;
      handlers.onStatus?.("reconnecting", { attempt, retryInMs: delay });
      timer = schedule(() => {
        timer = null;
        connect();
      }, delay);
    };
  };

  connect();

  return {
    close() {
      if (closed) return;
      closed = true;
      cleanup();
      handlers.onStatus?.("closed", { attempt: 0, retryInMs: null });
    },
    get closed() {
      return closed;
    },
  };
}

/** The shared client used by the app. Tests build their own with `createApiClient`. */
let sharedClient: ApiClient | null = null;

export function getApiClient(): ApiClient {
  sharedClient ??= createApiClient();
  return sharedClient;
}

export function resetApiClientForTests(): void {
  sharedClient = null;
}

export type { PluginManifest };

/** Normalise /api/runs/:id/artifacts items: objects from the server, or plain paths from older run records. */
export function parseArtifacts(list: readonly unknown[]): Artifact[] {
  return list
    .map((item): Artifact | null => {
      if (typeof item === 'string') return { path: item, bytes: 0, modified_at: '' };
      if (!item || typeof item !== 'object') return null;
      const record = item as Record<string, unknown>;
      if (typeof record.path !== 'string') return null;
      return {
        path: record.path,
        bytes: typeof record.bytes === 'number' ? record.bytes : 0,
        modified_at: typeof record.modified_at === 'string' ? record.modified_at : '',
      };
    })
    .filter((artifact): artifact is Artifact => artifact !== null);
}
