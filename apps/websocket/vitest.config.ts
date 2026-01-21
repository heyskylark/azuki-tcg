/// <reference types="vitest" />
import { defineConfig } from "vitest/config";
import path from "path";

export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    include: ["tests/**/*.test.ts"],
    setupFiles: ["tests/setup.ts"],
    testTimeout: 30000,
    hookTimeout: 30000,
    // Run tests sequentially by default for integration tests
    // that may share database state
    pool: "forks",
    poolOptions: {
      forks: {
        singleFork: true,
      },
    },
    // Retry failed tests once (useful for flaky network tests)
    retry: 1,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "@tcg/backend-core": path.resolve(__dirname, "../../packages/backend-core/dist"),
      "@test": path.resolve(__dirname, "./tests"),
    },
  },
});
