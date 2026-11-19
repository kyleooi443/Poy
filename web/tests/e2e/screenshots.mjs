// Regenerates docs/screenshots/*.png from the built site in real headless Chrome.
// Usage: npm run build && node tests/e2e/screenshots.mjs <hand.mjpeg>
// The fake webcam shows a still photo (any image saved as JPEG), so the "live" label is a demo.
import { mkdirSync } from "node:fs";
import { chromium } from "playwright-core";
import { preview } from "vite";

const out = new URL("../../../docs/screenshots/", import.meta.url).pathname;
mkdirSync(out, { recursive: true });
const hand = process.argv[2];
const server = await preview({ preview: { port: 4174, strictPort: true } });
const url = "http://localhost:4174/";
const FAKE = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"];

async function session(args, viewport) {
  const browser = await chromium.launch({ channel: "chrome", headless: true, args });
  const page = await browser.newPage({ viewport, deviceScaleFactor: 2 });
  await page.goto(url);
  return { browser, page };
}
const seeing = (p) =>
  p.waitForFunction(() => /^Seeing:/.test(document.getElementById("seeing").textContent), null, {
    timeout: 30000,
  });

{
  const { browser, page } = await session([...FAKE, `--use-file-for-fake-video-capture=${hand}`], {
    width: 900,
    height: 900,
  });
  await page.click("#start-camera");
  await seeing(page);
  await page.locator("body").focus();
  for (let i = 0; i < 3; i++) (await page.keyboard.press("Space"), await page.waitForTimeout(150));
  await page.screenshot({ path: `${out}play.png`, fullPage: true });

  await page.click('a[href="#train"]');
  for (const label of ["rock", "paper", "scissors"]) {
    await page.click(`[data-record="${label}"]`);
    await page.waitForTimeout(2300);
  }
  await page.click("#train-knn");
  await page.screenshot({ path: `${out}train.png`, fullPage: true });

  await page.click('a[href="#about"]');
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${out}about.png` }); // top of the page only: the attribution list is long
  await browser.close();
}
{
  const { browser, page } = await session([...FAKE, `--use-file-for-fake-video-capture=${hand}`], {
    width: 390,
    height: 844,
  });
  await page.click("#start-camera");
  await seeing(page);
  await page.screenshot({ path: `${out}mobile.png`, fullPage: true });
  await browser.close();
}
{
  const { browser, page } = await session(
    ["--use-fake-device-for-media-stream", "--deny-permission-prompts"],
    { width: 900, height: 700 },
  );
  await page.click("#start-camera");
  await page.waitForFunction(() => /denied/i.test(document.getElementById("status").textContent));
  await page.screenshot({ path: `${out}camera-denied.png` });
  await browser.close();
}
await server.close();
console.log("wrote screenshots to", out);
