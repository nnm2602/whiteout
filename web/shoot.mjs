// node shoot.mjs URL out.png [width height] [js-to-run-before-shot]
import { chromium } from 'playwright'
const [url, out, w = '1600', h = '950', js = ''] = process.argv.slice(2)
const browser = await chromium.launch({ executablePath: process.env.CHROME })
const page = await browser.newPage({ viewport: { width: +w, height: +h }, deviceScaleFactor: 1, colorScheme: 'light' })
await page.goto(url, { waitUntil: 'networkidle' })
await page.waitForTimeout(1200)
if (js) { await page.evaluate(js); await page.waitForTimeout(1500) }
await page.screenshot({ path: out })
await browser.close()
