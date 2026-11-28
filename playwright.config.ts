import { defineConfig } from "@playwright/test";

const BACKEND = "http://127.0.0.1:8001";
export const AGENT_STANDIN = "http://127.0.0.1:8003";

export default defineConfig({
  testDir: "e2e",
  outputDir: "e2e/test-results",
  reporter: "list",
  use: { baseURL: "http://localhost:5174" },
  webServer: [
    {
      command: "node e2e/agent-standin.js",
      url: `${AGENT_STANDIN}/health`,
      reuseExistingServer: false,
    },
    {
      command:
        "uv run -m backend.seed && uv run -m uvicorn backend.main:app --port 8001",
      url: `${BACKEND}/api/health`,
      env: { UNDERCOVER_DB: "e2e/e2e.db", UNDERCOVER_AGENT: AGENT_STANDIN },
      reuseExistingServer: false,
    },
    {
      command: "npx vite dashboard --port 5174 --strictPort",
      url: "http://localhost:5174",
      env: { BACKEND_URL: BACKEND },
      reuseExistingServer: false,
    },
  ],
});
