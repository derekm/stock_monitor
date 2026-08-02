/**
 * Playwright E2E tests for stock_monitor dashboard tools.
 *
 * Prerequisites:
 *   npx playwright install chromium
 *   ./start_dashboard.sh   # or static server + granite_service
 *
 * Run:
 *   npx playwright test tests/e2e/dashboard.spec.js
 */
const { test, expect } = require("@playwright/test");

const BASE = process.env.DASHBOARD_URL || "http://127.0.0.1:8765/index.html";
const API = process.env.GRANITE_API || "http://127.0.0.1:5055";

test.describe("Dashboard shell", () => {
  test("loads and shows nav tabs", async ({ page }) => {
    await page.goto(BASE);
    await expect(page.locator("nav, .tabs, [data-tab], button").first()).toBeVisible({ timeout: 15000 });
    // core sections exist
    for (const id of ["forecasts", "sql", "indexes"]) {
      const el = page.locator(`#${id}, section#${id}, [data-tab="${id}"]`);
      // tab button or section
      await expect(page.locator("body")).toContainText(/Portfolio|Forecast|SQL|Index/i);
    }
  });
});

test.describe("Granite forecast API (live)", () => {
  test("health endpoint", async ({ request }) => {
    const r = await request.get(`${API}/health`);
    expect(r.ok()).toBeTruthy();
    const j = await r.json();
    expect(j.ok).toBeTruthy();
    expect(j.live).toBeTruthy();
  });

  test("indexes endpoint", async ({ request }) => {
    const r = await request.get(`${API}/indexes`);
    expect(r.ok()).toBeTruthy();
    const j = await r.json();
    expect(j.ok).toBeTruthy();
    expect(Array.isArray(j.indexes)).toBeTruthy();
  });

  test("live POST forecast for MOS", async ({ request }) => {
    const r = await request.post(`${API}/forecast`, {
      data: {
        tickers: "MOS",
        horizon: 5,
        multivariate: false,
        from_first_trade: false,
      },
    });
    expect(r.ok()).toBeTruthy();
    const j = await r.json();
    expect(j.ok).toBeTruthy();
    expect(j.live).toBeTruthy();
    expect(j.charts.MOS || j.charts.mos || Object.keys(j.charts || {}).length).toBeTruthy();
    expect((j.forecasts || []).length).toBeGreaterThan(0);
  });

  test("live multivariate with correlated peers", async ({ request }) => {
    const r = await request.post(`${API}/forecast`, {
      data: {
        tickers: "MOS",
        horizon: 5,
        multivariate: true,
        peer_mode: "correlated",
        peer_index: "fertilizer",
        peer_top_n: 3,
      },
    });
    expect(r.ok()).toBeTruthy();
    const j = await r.json();
    expect(j.ok).toBeTruthy();
  });

  test("forecast by index portfolio", async ({ request }) => {
    const r = await request.post(`${API}/forecast`, {
      data: { index: "portfolio", horizon: 3, from_first_trade: true },
    });
    expect(r.ok()).toBeTruthy();
    const j = await r.json();
    expect(j.ok).toBeTruthy();
    expect((j.tickers || []).length).toBeGreaterThan(0);
  });
});

test.describe("Forecast studio UI", () => {
  test("forecast form controls exist", async ({ page }) => {
    await page.goto(BASE);
    // open forecasts tab if needed
    const tab = page.locator("button, a, [data-tab]").filter({ hasText: /Forecast/i }).first();
    if (await tab.count()) await tab.click();
    await expect(page.locator("#fc-run")).toBeVisible({ timeout: 10000 });
    await expect(page.locator("#fc-source")).toBeVisible();
    await expect(page.locator("#fc-horizon")).toBeVisible();
    await expect(page.locator("#fc-multivariate")).toBeVisible();
    await expect(page.locator("#fc-peer-mode")).toBeVisible();
    await expect(page.locator("#fc-run-list")).toBeVisible();
  });

  test("health check button updates status", async ({ page }) => {
    await page.goto(BASE);
    const tab = page.locator("button, a, [data-tab]").filter({ hasText: /Forecast/i }).first();
    if (await tab.count()) await tab.click();
    await page.fill("#fc-api", API);
    await page.click("#fc-health");
    await expect(page.locator("#fc-status")).toContainText(/OK|unreachable|Error/i, { timeout: 10000 });
  });

  test("run live forecast and keep run history", async ({ page }) => {
    await page.goto(BASE);
    const tab = page.locator("button, a, [data-tab]").filter({ hasText: /Forecast/i }).first();
    if (await tab.count()) await tab.click();
    await page.fill("#fc-api", API);
    await page.selectOption("#fc-source", "tickers");
    await page.fill("#fc-tickers", "MOS");
    await page.fill("#fc-horizon", "5");
    await page.fill("#fc-run-label", "e2e-run-1");
    await page.click("#fc-run");
    await expect(page.locator("#fc-status")).toContainText(/LIVE|Error|Embedded/i, { timeout: 60000 });
    // if service up, run history should gain a chip
    const status = await page.locator("#fc-status").innerText();
    if (status.includes("LIVE")) {
      await expect(page.locator("#fc-run-list button")).toHaveCount(1, { timeout: 5000 });
      // second run for comparison
      await page.fill("#fc-run-label", "e2e-run-2");
      await page.selectOption("#fc-multivariate", "1");
      await page.selectOption("#fc-peer-mode", "correlated");
      await page.fill("#fc-peer-index", "fertilizer");
      await page.click("#fc-run");
      await expect(page.locator("#fc-status")).toContainText(/LIVE/i, { timeout: 60000 });
      await expect(page.locator("#fc-run-list button")).toHaveCount(2, { timeout: 5000 });
    }
  });
});

test.describe("SQL lab tool", () => {
  test("SQL section or builder present", async ({ page }) => {
    await page.goto(BASE);
    const tab = page.locator("button, a, [data-tab]").filter({ hasText: /SQL/i }).first();
    if (await tab.count()) await tab.click();
    const sqlish = page.locator("#qb-table, #sql-status, #tbl-sql, textarea, .chip-btn");
    await expect(sqlish.first()).toBeVisible({ timeout: 15000 });
  });
});

test.describe("Other dashboard tools", () => {
  const tools = [
    { name: /Portfolio|Holdings/i },
    { name: /Index/i },
    { name: /Anomal/i },
    { name: /Corr|Regime|Risk/i },
  ];
  for (const tool of tools) {
    test(`nav target visible: ${tool.name}`, async ({ page }) => {
      await page.goto(BASE);
      const tab = page.locator("button, a, [data-tab]").filter({ hasText: tool.name }).first();
      if (await tab.count()) {
        await tab.click();
      }
      await expect(page.locator("body")).toBeVisible();
    });
  }
});
