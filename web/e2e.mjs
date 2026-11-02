// End-to-end check of the job page panels in a real browser.
// node e2e.mjs http://localhost:8091 JOB_ID out_dir voice.wav gauge.jpg
import { chromium } from 'playwright'
import fs from 'node:fs'

const [base, job, outDir, voice, gauge] = process.argv.slice(2)
fs.mkdirSync(outDir, { recursive: true })
const browser = await chromium.launch({
  executablePath: process.env.CHROME,
  args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-audio-capture=${voice}`],
})
const ctx = await browser.newContext({ viewport: { width: 1600, height: 950 }, permissions: ['microphone'] })
const page = await ctx.newPage()
const errors = []
page.on('pageerror', e => errors.push(String(e)))
page.on('console', m => { if (m.type() === 'error' || m.text().startsWith('listen:')) errors.push(m.text()) })
const results = {}
const shot = async name => page.screenshot({ path: `${outDir}/${name}.png` })
const step = async (name, fn) => {
  try { results[name] = (await fn()) ?? 'ok' } catch (e) { results[name] = `FAILED: ${String(e).slice(0, 300)}`; await shot(`${name}_failed`) }
}

await page.goto(`${base}/#/job/${job}`, { waitUntil: 'networkidle' })
await page.waitForSelector('.tabs.panels')

await step('plan', async () => {
  await page.click('.tabs.panels button:has-text("Plan")')
  await page.waitForSelector('.setup', { timeout: 60000 })
  const n = await page.locator('.setup').count()
  await page.click('.setuphead:has-text("CMM")')
  await page.waitForSelector('table.plan tr')
  await page.locator('table.plan tr').first().click()
  await page.waitForTimeout(800)
  await shot('plan')
  const dmi = await page.evaluate(async j => (await fetch(`/api/jobs/${j}/program.dmi`)).text(), job)
  if (!dmi.includes('ENDFIL')) throw new Error('DMIS program has no ENDFIL')
  return `${n} setups; DMIS ${dmi.split('\n').length} lines`
})

await step('measure_typed', async () => {
  await page.click('.tabs.panels button:has-text("Measure")')
  await page.waitForSelector('.stepcard')
  // walk to a line with numeric limits
  for (let k = 0; k < 60; k++) {
    const lim = await page.locator('.lim.big').innerText()
    if (/\d/.test(lim)) break
    await page.click('button:has-text("Skip")')
    await page.waitForTimeout(120)
  }
  const rid = await page.locator('.bignum').innerText()
  const lim = await page.locator('.lim.big').innerText()
  const lo = parseFloat(lim.split('–')[0])
  await page.fill('.entry input', String(lo + 0.0001))
  await page.press('.entry input', 'Enter')
  await page.waitForTimeout(800)
  await page.click('button:has-text("Back")')
  await page.waitForTimeout(300)
  const st = await page.locator('.entry .st').innerText()
  await page.click('button:has-text("Skip")')
  return `line ${rid} limits ${lim} typed ${lo + 0.0001}: ${st}`
})

await step('measure_voice', async () => {
  const rid = await page.locator('.bignum').innerText()
  await page.click('button:has-text("Say it")')
  await page.waitForFunction(() => /Heard|nothing|not available|limit/.test(document.querySelector('.measure > .fine')?.textContent || ''), null, { timeout: 40000 })
  const msg = await page.locator('.measure > p.fine').first().innerText()
  await shot('measure_voice')
  if (!msg.includes('Heard “')) throw new Error(msg)
  return `line ${rid}: ${msg}`
})

await step('measure_photo', async () => {
  await page.click('button:has-text("Save and next")')
  await page.waitForTimeout(400)
  await page.setInputFiles('.entry input[type=file]', gauge)
  await page.waitForSelector('.photo', { timeout: 90000 })
  await page.waitForTimeout(500)
  await shot('measure_photo')
  return await page.locator('.photo').innerText()
})

await step('ask', async () => {
  await page.click('.tabs.panels button:has-text("Ask")')
  await page.click('.examples button >> nth=2')
  await page.waitForSelector('.qa', { timeout: 90000 })
  await page.waitForTimeout(600)
  await shot('ask')
  const a = await page.locator('.qa').first().innerText()
  const stack = await page.locator('table.stack').count()
  return `${a.slice(0, 200)} | stack tables: ${stack}`
})

await step('supplier', async () => {
  await page.click('.tabs.panels button:has-text("Supplier")')
  await page.click('button:has-text("Try: Supplier FAIR (spreadsheet).xlsx")')
  await page.waitForSelector('.verdict', { timeout: 60000 })
  await page.locator('.finding').first().click()
  await page.waitForTimeout(800)
  await shot('supplier')
  return `${await page.locator('.verdict').innerText()} | findings ${await page.locator('.finding').count()}`
})

await step('model', async () => {
  await page.click('.tabs.panels button:has-text("3D model")')
  await page.locator('button:has-text("Try:")').first().click()
  await page.waitForSelector('.check .chips', { timeout: 60000 })
  const f = page.locator('.finding')
  if (await f.count()) await f.first().click()
  await page.waitForTimeout(800)
  await shot('model')
  const dmi = await page.evaluate(async j => (await fetch(`/api/jobs/${j}/program.dmi`)).text(), job)
  const located = dmi.split('\n').filter(l => /FEAT\/CIRCLE/.test(l) && !/CART,0,0,0/.test(l)).length
  return `${await page.locator('.check .chips').innerText()} | CMM holes with model positions: ${located}`
})

await step('form3_after', async () => {
  await page.click('.tabs.panels button:has-text("Form 3")')
  await page.waitForSelector('.form3 tr')
  const filled = await page.evaluate(async j => (await (await fetch(`/api/jobs/${j}`)).json()).rows.filter(r => r.measured).length, job)
  await shot('form3')
  return `${filled} lines with a measured value`
})

results.page_errors = errors
fs.writeFileSync(`${outDir}/results.json`, JSON.stringify(results, null, 1))
console.log(JSON.stringify(results, null, 1))
await browser.close()
