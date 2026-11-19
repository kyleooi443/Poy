// Browser smoke test: real Chrome, real MediaPipe, fake webcam fed from a still image.
// Usage: npm run build && node tests/e2e/smoke.mjs [hand.mjpeg]
// Not part of `make test` (needs Google Chrome installed and takes ~30 s).
import assert from "node:assert/strict";
import { chromium } from "playwright-core";
import { preview } from "vite";

const handVideo = process.argv[2];
const server = await preview({ preview: { port: 4173, strictPort: true } });
const url = "http://localhost:4173/";
const results = [];

/** @param {string[]} args */
async function open(args, setup = async () => {}) {
  const browser = await chromium.launch({ channel: "chrome", headless: true, args });
  const page = await browser.newPage({ acceptDownloads: true });
  const external = [];
  page.on("request", (r) => {
    if (!r.url().startsWith(url)) external.push(r.url());
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await setup(page);
  await page.goto(url);
  return { browser, page, external, errors };
}

async function check(name, fn) {
  try {
    await fn();
    results.push(`PASS ${name}`);
  } catch (e) {
    results.push(`FAIL ${name}: ${e.message.split("\n")[0]}`);
  }
}

const FAKE = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"];

await check("missing model.json shows a clear message", async () => {
  const { browser, page } = await open(FAKE, (p) =>
    p.route("**/models/model.json", (r) => r.fulfill({ status: 404, body: "" })),
  );
  await page.waitForFunction(() =>
    /model file missing/i.test(document.getElementById("status").textContent),
  );
  await browser.close();
});

await check("phone width: no horizontal scroll; Tab shows a visible focus ring", async () => {
  const { browser, page } = await open(FAKE);
  for (const hash of ["#play", "#train", "#about"]) {
    await page.setViewportSize({ width: 360, height: 740 });
    await page.goto(url + hash);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
    assert.ok(overflow <= 0, `${hash} overflows by ${overflow}px`);
  }
  await page.keyboard.press("Tab");
  await page.keyboard.press("Tab");
  const outline = await page.evaluate(() => getComputedStyle(document.activeElement).outlineStyle);
  assert.equal(outline, "solid");
  await browser.close();
});

await check("camera permission denied shows a clear message", async () => {
  const { browser, page } = await open([
    "--use-fake-device-for-media-stream",
    "--deny-permission-prompts",
  ]);
  await page.click("#start-camera");
  await page.waitForFunction(
    () => /denied/i.test(document.getElementById("status").textContent),
    null,
    {
      timeout: 10000,
    },
  );
  await browser.close();
});

await check(
  "fake camera without a hand: tracker loads, says no hand, Space refuses to play",
  async () => {
    const { browser, page, external, errors } = await open(FAKE);
    await page.click("#start-camera");
    await page.waitForFunction(
      () => document.getElementById("seeing").textContent === "No hand visible",
      null,
      {
        timeout: 30000,
      },
    );
    await page.locator("body").focus();
    await page.keyboard.press("Space");
    assert.match(await page.textContent("#status"), /can't see your hand/);
    assert.equal(await page.textContent("#score-player"), "0");
    assert.deepEqual(external, [], "no requests may leave the site");
    assert.deepEqual(errors, []);
    await browser.close();
  },
);

if (handVideo) {
  await check(
    "hand image as webcam: seen, round played with Space, train + download work",
    async () => {
      const { browser, page, external, errors } = await open([
        ...FAKE,
        `--use-file-for-fake-video-capture=${handVideo}`,
      ]);
      await page.click("#start-camera");
      await page.waitForFunction(
        () => /^Seeing:/.test(document.getElementById("seeing").textContent),
        null,
        {
          timeout: 30000,
        },
      );
      results.push(`  info: ${await page.textContent("#seeing")}`);
      await page.locator("body").focus();
      await page.keyboard.press("Space");
      const status = await page.textContent("#status");
      const result = await page.textContent("#result");
      if (status) results.push(`  info: status after Space: ${status}`);
      else {
        assert.match(result, /^You: (rock|paper|scissors)\. Computer: (rock|paper|scissors)\./);
        const total = await page.$$eval(".score dd", (dds) =>
          dds.reduce((a, d) => a + Number(d.textContent), 0),
        );
        assert.equal(total, 1);
        results.push(`  info: ${result}`);
      }

      await page.click('a[href="#train"]');
      for (const label of ["rock", "paper", "scissors"]) {
        await page.click(`[data-record="${label}"]`);
        await page.waitForTimeout(2300);
        assert.ok(
          Number(await page.textContent(`[data-count="${label}"]`)) > 0,
          `${label} samples`,
        );
      }
      await page.click("#train-knn");
      assert.match(await page.textContent("#status"), /Trained on \d+ samples/);
      const [download] = await Promise.all([
        page.waitForEvent("download"),
        page.click("#download-samples"),
      ]);
      const doc = JSON.parse(
        await (await download.createReadStream()).toArray().then((c) => Buffer.concat(c)),
      );
      assert.equal(doc.format, "poy-samples");
      assert.equal(doc.samples[0].features.length, 63);
      results.push(`  info: downloaded ${doc.samples.length} samples`);
      assert.deepEqual(external, [], "no requests may leave the site");
      assert.deepEqual(errors, []);
      await browser.close();
    },
  );
}

await server.close();
console.log(results.join("\n"));
process.exit(results.some((r) => r.startsWith("FAIL")) ? 1 : 0);
