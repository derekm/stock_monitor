// @ts-check
const { defineConfig } = require("@playwright/test");

module.exports = defineConfig({
  testDir: "./tests/e2e",
  timeout: 90_000,
  retries: 0,
  use: {
    headless: true,
    baseURL: process.env.DASHBOARD_URL || "http://127.0.0.1:8765",
  },
  reporter: [["list"]],
});
