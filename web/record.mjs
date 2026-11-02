// Screen recordings of the real app for the demo video.
// node record.mjs <scene> <baseUrl> <outDir>
import { chromium } from 'playwright'
import fs from 'node:fs'
import path from 'node:path'

const [scene, base, outDir = '../out/rec'] = process.argv.slice(2)
const CHROME = process.env.CHROME || `${process.env.HOME}/Library/Caches/ms-playwright/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing`
const W = 1920, H = 1080
const voice = path.resolve('../out/voice/dictate.wav')
const browser = await chromium.launch({
  executablePath: CHROME,
  args: ['--window-size=1920,1080', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-audio-capture=${voice}`],
})
const ctx = await browser.newContext({ viewport: { width: W, height: H }, colorScheme: 'light', recordVideo: { dir: outDir, size: { width: W, height: H } } })
await ctx.grantPermissions(['microphone'])
const page = await ctx.newPage()
const wait = ms => page.waitForTimeout(ms)
const t0 = Date.now()
const marks = []
const mark = label => marks.push({ t: (Date.now() - t0) / 1000, label })

async function job(id) {
  await page.goto(`${base}/#/job/${id}`, { waitUntil: 'networkidle' })
  await wait(1500)
}

if (scene === 'erase') { // Apollo sample: balloons, then the whiteout replay
  await job(process.env.JOB)
  mark('balloons'); await wait(3000)
  await page.click('text=Whiteout check'); await wait(800)
  mark('replay'); await page.click('text=Replay the erase'); await wait(9000)
  mark('residue'); await wait(4000)
} else if (scene === 'upload') { // fresh upload, live reading
  await page.goto(base + '/', { waitUntil: 'networkidle' }); await wait(1500)
  mark('home')
  await page.setInputFiles('#file', process.env.FILE)
  mark('uploaded')
  await page.waitForURL(/#\/job\//)
  await page.waitForFunction(() => document.querySelector('.actions') !== null, null, { timeout: 900000 })
  mark('done'); await wait(4000)
} else if (scene === 'form3') { // pick a row, type an out-of-tolerance value
  await job(process.env.JOB)
  await page.click('.tabs.small button:nth-child(2)').catch(() => {}); await wait(1500)
  const row = page.locator('tbody tr', { hasText: process.env.ROWTEXT || '2.375' }).first()
  mark('row'); await row.click(); await wait(2500)
  const input = row.locator('input')
  mark('type'); await input.click(); await input.fill(''); await input.type(process.env.VALUE || '2.3712', { delay: 140 }); await wait(2500)
} else if (scene === 'voice') { // hold to dictate, real Muse Voice Transcribe call
  await job(process.env.JOB)
  const btn = page.locator('text=Hold to dictate results')
  const box = await btn.boundingBox()
  mark('hold'); await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2); await page.mouse.down()
  await wait(+(process.env.HOLD || 6500))
  mark('release'); await page.mouse.up()
  await page.waitForFunction(() => /Heard/.test(document.querySelector('.actions')?.textContent || ''), null, { timeout: 60000 })
  mark('heard'); await wait(4000)
} else if (scene === 'revision') {
  await job(process.env.JOB)
  await page.click('text=Changes'); await wait(600)
  await page.click('.tabs.small button:nth-child(2)'); await wait(1200)
  mark('changes'); await wait(5000)
}
fs.writeFileSync(path.join(outDir, `${scene}.marks.json`), JSON.stringify(marks, null, 1))
const v = page.video()
await ctx.close()
const p = await v.path()
fs.renameSync(p, path.join(outDir, `${scene}.webm`))
await browser.close()
console.log('saved', path.join(outDir, `${scene}.webm`), marks)
