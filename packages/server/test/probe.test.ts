import { createRequire } from "node:module";
import { RunSpecSchema } from "@supertuner/core";
import { describe, expect, it } from "vitest";

describe("probe", () => {
  it("loads node:sqlite through createRequire", () => {
    const require = createRequire(import.meta.url);
    const sqlite = require("node:sqlite") as { DatabaseSync: new (path: string) => { exec(sql: string): void; prepare(sql: string): { get(): unknown }; close(): void } };
    const db = new sqlite.DatabaseSync(":memory:");
    db.exec("create table t(a text)");
    expect(db.prepare("select 1 as n").get()).toBeTruthy();
    db.close();
    expect(RunSpecSchema.safeParse({ config: { name: "x", family: "y", stage: "sft" } }).success).toBe(true);
  });
});