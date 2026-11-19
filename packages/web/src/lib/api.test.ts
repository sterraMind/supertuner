import { describe, expect, it, vi } from "vitest";
import { ApiError, apiOrigin, apiUrl, createApiClient, formatApiError } from "./api";
import { defaultRunSpec } from "./planner-state";

function fakeResponse(body: unknown, status = 200): Response {
  const text = JSON.stringify(body);
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => text,
    json: async () => body,
  } as unknown as Response;
}

describe("apiUrl", () => {
  it("appends the /api prefix to a bare origin", () => {
    expect(apiUrl("/health", "http://127.0.0.1:8787")).toBe("http://127.0.0.1:8787/api/health");
    expect(apiUrl("health", "http://127.0.0.1:8787")).toBe("http://127.0.0.1:8787/api/health");
  });

  it("does not double the prefix when the origin already ends in /api", () => {
    expect(apiUrl("/health", "http://127.0.0.1:8787/api")).toBe("http://127.0.0.1:8787/api/health");
    expect(apiUrl("/api/health", "http://127.0.0.1:8787")).toBe("http://127.0.0.1:8787/api/health");
  });

  it("falls back to a same-origin relative path", () => {
    expect(apiOrigin("")).toBe("");
    expect(apiUrl("/runs", "")).toBe("/api/runs");
  });

  it("trims trailing slashes from the origin", () => {
    expect(apiOrigin("http://example.test/// ")).toBe("http://example.test");
  });
});

describe("formatApiError", () => {
  it("passes through plain error messages", () => {
    expect(formatApiError(new Error("boom"))).toContain("boom");
    expect(formatApiError("plain text")).toContain("plain text");
  });

  it("includes the first issue of an ApiError", () => {
    const error = new ApiError("Run rejected", {
      status: 422,
      issues: [{ severity: "error", code: "memory.vram", message: "Training peak exceeds the budget", hint: "Lower the batch size", field: null }],
    });
    const formatted = formatApiError(error);
    expect(formatted).toContain("Run rejected");
    expect(formatted).toContain("Training peak exceeds the budget");
  });

  it("handles unknown thrown values", () => {
    expect(formatApiError(undefined)).toBeTruthy();
    expect(formatApiError({ weird: true })).toBeTruthy();
  });
});

describe("createApiClient error handling", () => {
  it("turns a 422 payload into an ApiError with issues and the server message", async () => {
    const fetchImpl = vi.fn(async () =>
      fakeResponse(
        {
          error: "Run rejected by the compiler",
          issues: [{ severity: "error", code: "memory.vram", message: "peak too high", hint: "" }],
        },
        422,
      ),
    );
    const client = createApiClient({ baseUrl: "http://127.0.0.1:8787", fetchImpl: fetchImpl as unknown as typeof fetch });

    await expect(client.createRun(defaultRunSpec())).rejects.toMatchObject({
      name: "ApiError",
      status: 422,
      message: "Run rejected by the compiler",
    });

    const error = await client.createRun(defaultRunSpec()).catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(ApiError);
    const apiError = error as ApiError;
    expect(apiError.issues).toHaveLength(1);
    expect(apiError.issues[0]?.code).toBe("memory.vram");
    expect(apiError.url).toContain("/api/runs");
  });

  it("reports schema violations instead of trusting the payload", async () => {
    const fetchImpl = vi.fn(async () => fakeResponse([{ id: "not-a-recipe" }]));
    const client = createApiClient({ baseUrl: "http://127.0.0.1:8787", fetchImpl: fetchImpl as unknown as typeof fetch });

    const error = await client.recipes().catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(ApiError);
    const apiError = error as ApiError;
    expect(apiError.issues.length).toBeGreaterThan(0);
    expect(apiError.message).toContain("schema");
    expect(apiError.url).toContain("/api/recipes");
  });

  it("wraps network failures with status 0", async () => {
    const fetchImpl = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    });
    const client = createApiClient({ baseUrl: "http://127.0.0.1:8787", fetchImpl: fetchImpl as unknown as typeof fetch });

    const error = await client.health().catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(0);
    expect((error as ApiError).message).toContain("Failed to fetch");
  });
});