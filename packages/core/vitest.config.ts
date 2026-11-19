import { defineConfig } from "vitest/config";

/**
 * 跑 packages/core 的契约测试。测试直接 import src 源码，
 * 不需要 alias（测试文件用相对路径 ../src 引用）。
 */
export default defineConfig({
  test: {
    environment: "node",
    include: ["test/**/*.test.ts"],
    testTimeout: 30_000,
    hookTimeout: 30_000,
  },
});
