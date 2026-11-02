/** Push-to-talk recording to a 16 kHz mono 16-bit WAV, the format Muse Voice Transcribe takes. */
export class Recorder {
  private ctx?: AudioContext
  private stream?: MediaStream
  private node?: ScriptProcessorNode
  private chunks: Float32Array[] = []

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true } })
    this.ctx = new AudioContext()
    const src = this.ctx.createMediaStreamSource(this.stream)
    this.node = this.ctx.createScriptProcessor(4096, 1, 1)
    this.chunks = []
    this.node.onaudioprocess = e => this.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)))
    src.connect(this.node)
    this.node.connect(this.ctx.destination)
  }

  async stop(): Promise<Blob> {
    const rate = this.ctx?.sampleRate ?? 48000
    this.node?.disconnect()
    this.stream?.getTracks().forEach(t => t.stop())
    await this.ctx?.close()
    const len = this.chunks.reduce((a, c) => a + c.length, 0)
    const all = new Float32Array(len)
    let o = 0
    for (const c of this.chunks) { all.set(c, o); o += c.length }
    return toWav(resample(all, rate, 16000), 16000)
  }
}

function resample(x: Float32Array, from: number, to: number): Float32Array {
  if (from === to) return x
  const n = Math.floor((x.length * to) / from)
  const out = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const t = (i * from) / to
    const j = Math.floor(t)
    const f = t - j
    out[i] = (x[j] ?? 0) * (1 - f) + (x[j + 1] ?? 0) * f
  }
  return out
}

function toWav(x: Float32Array, rate: number): Blob {
  const buf = new ArrayBuffer(44 + x.length * 2)
  const v = new DataView(buf)
  const w = (o: number, s: string) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)))
  w(0, 'RIFF'); v.setUint32(4, 36 + x.length * 2, true); w(8, 'WAVE'); w(12, 'fmt ')
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, rate, true)
  v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, 'data')
  v.setUint32(40, x.length * 2, true)
  for (let i = 0; i < x.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, x[i])) * 0x7fff, true)
  return new Blob([buf], { type: 'audio/wav' })
}

/** Hands-free: listen until the inspector has spoken and then paused, or give up after a while. */
export async function listenOnce(opts: { maxMs?: number; quietMs?: number; startMs?: number; onLevel?: (v: number) => void } = {}): Promise<Blob | null> {
  const { maxMs = 9000, quietMs = 900, startMs = 7000, onLevel } = opts
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false, autoGainControl: true } })
  const ctx = new AudioContext()
  await ctx.resume()
  const src = ctx.createMediaStreamSource(stream)
  const node = ctx.createScriptProcessor(4096, 1, 1)
  const chunks: Float32Array[] = []
  let peak = 0
  const t0 = performance.now()
  let heard = 0
  let lastLoud = 0
  let floor = 1
  return new Promise(resolve => {
    const finish = async (keep: boolean) => {
      node.disconnect(); stream.getTracks().forEach(t => t.stop())
      const rate = ctx.sampleRate
      await ctx.close()
      if (!keep) return resolve(null)
      const len = chunks.reduce((a, c) => a + c.length, 0)
      const all = new Float32Array(len)
      let o = 0
      for (const c of chunks) { all.set(c, o); o += c.length }
      resolve(toWav(resample(all, rate, 16000), 16000))
    }
    node.onaudioprocess = e => {
      const x = e.inputBuffer.getChannelData(0)
      chunks.push(new Float32Array(x))
      let s = 0
      for (let i = 0; i < x.length; i++) s += x[i] * x[i]
      const rms = Math.sqrt(s / x.length)
      onLevel?.(rms)
      peak = Math.max(peak, rms)
      const now = performance.now() - t0
      floor = Math.max(0.002, Math.min(floor, rms)) // the quietest moment so far is the room
      if (now < 250) return
      if (rms > Math.max(0.015, floor * 3)) { if (!heard) heard = now; lastLoud = now }
      if (heard && now - lastLoud > quietMs && now - heard > 350) finish(true)
      else if (!heard && now > startMs) { console.info(`listen: nothing above the room level (peak ${peak.toFixed(3)}, floor ${floor.toFixed(3)})`); finish(false) }
      else if (now > maxMs) finish(!!heard)
    }
    src.connect(node)
    node.connect(ctx.destination)
  })
}

/** Read a line aloud with the browser's own voice (Meta's API has no text to speech). */
export function say(text: string): Promise<void> {
  return new Promise(resolve => {
    if (!('speechSynthesis' in window)) return resolve()
    speechSynthesis.cancel()
    const u = new SpeechSynthesisUtterance(text)
    u.rate = 1.05
    u.onend = () => resolve()
    u.onerror = () => resolve()
    speechSynthesis.speak(u)
  })
}

/** "Ø.182 - .192" -> words a person would say. */
export function spoken(s: string): string {
  return s.replace(/[Ø⌀∅]/g, ' diameter ').replace(/↧/g, ' depth ').replace(/⊔/g, ' counterbore ').replace(/±/g, ' plus or minus ')
    .replace(/(\d*)\.(\d+)/g, (_, a: string, b: string) => `${a ? a + ' ' : ''}point ${b.split('').join(' ')}`)
    .replace(/\s-\s/g, ' to ').replace(/\s+/g, ' ').trim()
}
