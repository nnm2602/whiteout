import { useEffect, useMemo, useRef, useState } from 'react'
import { getJSON, putResults, type Job, type Row } from './api'
import { listenOnce, say, spoken } from './voice'

export interface PlanStep {
  row_id: string
  char_id: string
  number: string
  sheet: number
  zone: string
  requirement: string
  type: string
  lower: string
  upper: string
  datums: string[]
  setup: string
  instrument: string
  method: string
  why: string
}

export interface Plan {
  units: string
  steps: PlanStep[]
  setups: { setup: string; instruments: string[]; rows: string[]; count: number }[]
  to_measure: number
  instruments: string[]
}

type Focus = (charId: string) => void

const limits = (s: { lower: string; upper: string }) => (s.lower || s.upper ? `${s.lower} – ${s.upper}` : '')

function usePlan(job: Job) {
  const [plan, setPlan] = useState<Plan | null>(null)
  const [err, setErr] = useState('')
  useEffect(() => { getJSON<Plan>(`/api/jobs/${job.id}/plan`).then(setPlan).catch(e => setErr(String(e.message))) }, [job.id])
  return { plan, err }
}

/* ------------------------------------------------------------------ measuring plan */
export function PlanPanel({ job, onFocus, onMeasure }: { job: Job; onFocus: Focus; onMeasure: () => void }) {
  const { plan, err } = usePlan(job)
  const [open, setOpen] = useState<string>('')
  if (err) return <p className="err">{err}</p>
  if (!plan) return <p className="fine pulse">Working out an instrument and a method for every line…</p>
  const by = Object.fromEntries(plan.steps.map(s => [s.row_id, s]))
  return (
    <div className="panel">
      <div className="panelhead">
        <div>
          <h3>Measuring plan</h3>
          <p className="fine">{plan.to_measure} lines to measure in {plan.setups.filter(g => g.setup !== 'Not measured').length} setups, ordered so the part is picked up as few times as possible. Instruments are chosen in code from each tolerance band; Muse Spark sorts the notes.</p>
        </div>
        <div className="row">
          <button className="btn primary" onClick={onMeasure}>Start measuring</button>
          <a className="btn" href={`/api/jobs/${job.id}/program.dmi`}>CMM program (.dmi)</a>
        </div>
      </div>
      {plan.setups.map(g => (
        <div key={g.setup} className={`setup ${g.setup === 'Not measured' ? 'muted' : ''}`}>
          <button className="setuphead" onClick={() => setOpen(open === g.setup ? '' : g.setup)}>
            <b>{g.setup}</b><span className="count">{g.count}</span>
            <span className="fine">{g.instruments.slice(0, 4).join(' · ')}{g.instruments.length > 4 ? ' …' : ''}</span>
          </button>
          {open === g.setup && (
            <table className="plan">
              <tbody>
                {g.rows.map(r => by[r]).map(s => (
                  <tr key={s.row_id} onClick={() => onFocus(s.char_id)}>
                    <td className="num">{s.row_id}</td>
                    <td><span className="rq">{s.requirement}</span><span className="lbl">{limits(s)}</span></td>
                    <td><b>{s.instrument || 'none'}</b><span className="lbl">{s.method}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      ))}
    </div>
  )
}

/* ------------------------------------------------------------------ guided measuring */
interface Reading {
  display: string
  unit: string
  sure: boolean
  note: string
  value_text: string
  code_reading: string
  written: boolean
}

export function MeasurePanel({ job, onFocus, onRows }: { job: Job; onFocus: Focus; onRows: (r: Row[]) => void }) {
  const { plan, err } = usePlan(job)
  const steps = useMemo(() => plan?.steps.filter(s => s.setup !== 'Not measured') ?? [], [plan])
  const [i, setI] = useState(() => 0)
  const [msg, setMsg] = useState('')
  const [busy, setBusy] = useState('')
  const [handsFree, setHandsFree] = useState(false)
  const [level, setLevel] = useState(0)
  const [photo, setPhoto] = useState<{ url: string; r: Reading } | null>(null)
  const [draft, setDraft] = useState('')
  const hf = useRef(false)
  const step = steps[i]
  const row = job.rows.find(r => r.row_id === step?.row_id)

  useEffect(() => { if (step) { onFocus(step.char_id); setDraft(row?.measured ?? ''); setPhoto(null) } }, [step?.row_id]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => { hf.current = false; speechSynthesis?.cancel() }, [])

  if (err) return <p className="err">{err}</p>
  if (!plan) return <p className="fine pulse">Loading the measuring plan…</p>
  if (!step) return <p className="fine">Nothing to measure on this drawing.</p>

  const done = steps.filter(s => job.rows.find(r => r.row_id === s.row_id)?.measured).length
  const prompt = (s: PlanStep) => `Balloon ${s.row_id.replace('.', ' point ').replace('-', ', instance ')}. ${spoken(s.requirement)}. ${s.instrument}.`

  const write = async (v: string) => {
    const r = await putResults(job.id, { [step.row_id]: v })
    onRows(r.rows)
    const nr = r.rows.find(x => x.row_id === step.row_id)
    return nr
  }

  const listen = async (auto = false): Promise<boolean> => {
    setBusy('listening'); setMsg('Listening… say the value, for example "point one eight eight five"')
    try {
      const wav = await listenOnce({ onLevel: v => setLevel(Math.min(1, v * 12)) })
      setLevel(0)
      if (!wav) { setMsg('Heard nothing.'); return false }
      setBusy('transcribing'); setMsg('Muse Voice Transcribe…')
      const fd = new FormData(); fd.append('audio', wav, 'value.wav'); fd.append('row', step.row_id)
      const r = await fetch(`/api/jobs/${job.id}/voice`, { method: 'POST', body: fd }).then(r => r.json())
      if (r.error) { setMsg(r.error); return false }
      onRows(r.rows)
      const v = r.entries[0]?.[1]
      const nr = (r.rows as Row[]).find(x => x.row_id === step.row_id)
      if (!v) { setMsg(`Heard “${r.transcript}”: no value in it.`); return false }
      setDraft(v)
      setMsg(`Heard “${r.transcript}” → ${v} ${nr?.status ? `(${nr.status})` : ''}`)
      if (auto) await say(`${spoken(v)}. ${nr?.status === 'fail' ? 'Out of tolerance.' : nr?.status === 'pass' ? 'Good.' : ''}`)
      return true
    } catch { setMsg('Microphone not available'); return false } finally { setBusy(''); setLevel(0) }
  }

  const runHandsFree = async () => {
    hf.current = true; setHandsFree(true)
    let k = i
    while (hf.current && k < steps.length) {
      setI(k)
      await say(prompt(steps[k]))
      if (!hf.current) break
      const ok = await listen(true)
      if (!hf.current) break
      if (!ok) { await say('I did not get a value. Hands-free is paused.'); break }
      k += 1
    }
    hf.current = false; setHandsFree(false)
  }

  const shoot = async (f?: File) => {
    if (!f) return
    setBusy('photo'); setMsg('Muse Spark is reading the display…')
    const url = URL.createObjectURL(f)
    try {
      const fd = new FormData(); fd.append('photo', f); fd.append('row', step.row_id)
      const r = await fetch(`/api/jobs/${job.id}/gauge`, { method: 'POST', body: fd }).then(r => r.json())
      if (r.error) { setMsg(r.error); return }
      onRows(r.rows)
      setPhoto({ url, r: r.reading })
      if (r.reading.written) { setDraft(r.reading.value_text); setMsg(`Read ${r.reading.display} ${r.reading.unit}; both readers agree, written to line ${step.row_id}.`) }
      else setMsg('Not sure of this reading: check the photo and pick the value.')
    } finally { setBusy('') }
  }

  const status = row?.status
  return (
    <div className="panel measure">
      <div className="panelhead">
        <div>
          <h3>Guided measuring</h3>
          <p className="fine">Line {i + 1} of {steps.length} · {done} measured · setup: {step.setup}</p>
        </div>
        <div className="row">
          {handsFree
            ? <button className="btn rec" onClick={() => { hf.current = false; setHandsFree(false); speechSynthesis?.cancel() }}>■ Stop hands-free</button>
            : <button className="btn primary" onClick={runHandsFree} disabled={!!busy}>Hands-free</button>}
        </div>
      </div>
      <div className="stepcard">
        <div className="bignum">{step.row_id}</div>
        <div className="stepbody">
          <div className="rq big">{step.requirement}</div>
          <div className="lim big">{limits(step) || (row?.attribute ? 'attribute: OK or NC' : '')}</div>
          <div className="inst"><b>{step.instrument}</b> · {step.method}</div>
          <p className="fine">Sheet {step.sheet + 1}, zone {step.zone}{step.datums.length ? ` · datums ${step.datums.join('|')}` : ''}</p>
        </div>
      </div>
      <div className="entry">
        <input value={draft} placeholder={row?.attribute ? 'OK / NC' : 'measured value'} onChange={e => setDraft(e.target.value)}
          onKeyDown={async e => { if (e.key === 'Enter') { await write(draft); setI(Math.min(steps.length - 1, i + 1)) } }} />
        {status && <span className={`st ${status.replace(' ', '-')}`}>{status}</span>}
        <button className="btn" onClick={() => listen()} disabled={!!busy}>{busy === 'listening' ? <span className="meter" style={{ ['--lvl' as string]: level }}>● listening</span> : '🎙 Say it'}</button>
        <label className="btn">📷 Photo of the gauge<input type="file" accept="image/*" capture="environment" hidden onChange={e => shoot(e.target.files?.[0])} /></label>
      </div>
      {photo && (
        <div className="photo">
          <img src={photo.url} alt="gauge" />
          <div>
            <p><b>Muse Spark read {photo.r.display || 'nothing'}</b> {photo.r.unit !== 'unknown' ? photo.r.unit : ''}{photo.r.code_reading ? ` · the second reader (code) saw ${photo.r.code_reading}` : ''}</p>
            {photo.r.note && <p className="fine">{photo.r.note}</p>}
            {!photo.r.written && photo.r.value_text && (
              <button className="btn" onClick={async () => { setDraft(photo.r.value_text); await write(photo.r.value_text); setMsg(`Written: ${photo.r.value_text}`) }}>Use {photo.r.value_text}</button>
            )}
          </div>
        </div>
      )}
      <div className="row nav">
        <button className="btn" onClick={() => setI(Math.max(0, i - 1))} disabled={i === 0}>← Back</button>
        <button className="btn" onClick={async () => { if (draft !== (row?.measured ?? '')) await write(draft); setI(Math.min(steps.length - 1, i + 1)) }}>Save and next →</button>
        <button className="btn" onClick={() => setI(Math.min(steps.length - 1, i + 1))}>Skip</button>
      </div>
      {msg && <p className="fine">{msg}</p>}
      <p className="fine">Steps are read aloud by the browser's voice. Values are heard by Muse Voice Transcribe and checked against the limits in code. A gauge photo is written only when Muse Spark and a second reader in code agree; otherwise you choose.</p>
    </div>
  )
}

/* ------------------------------------------------------------------ ask the drawing */
interface Answer {
  q: string
  a: string
  rows: string[]
  chars: string[]
  stackup?: {
    what: string
    links: { row_id: string; sign: number; requirement: string; lower: string; upper: string }[]
    given: { what: string; value: string; sign: number }[]
    missing: string[]
    worst_case: { min: string; max: string }
    nominal: string
    rss: { min: string; max: string; plus_minus: string }
  }
}

const EXAMPLES = ['Which requirement has the tightest tolerance?', 'Which lines refer to datum B?', 'If the part is exactly .500 thick, what is the worst-case wall under counterbore 17?', 'Which lines are out of tolerance so far?']

export function AskPanel({ job, onFocus, onHighlight }: { job: Job; onFocus: Focus; onHighlight: (ids: string[]) => void }) {
  const [qs, setQs] = useState<Answer[]>([])
  const [q, setQ] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const ask = async (text: string) => {
    if (!text.trim()) return
    setBusy(true); setErr('')
    try {
      const r = await fetch(`/api/jobs/${job.id}/ask`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: text, history: qs.map(x => ({ q: x.q, a: x.a })) }),
      }).then(r => r.json())
      if (r.error) { setErr(r.error); return }
      setQs(l => [...l, { q: text, a: r.answer, rows: r.rows, chars: r.chars, stackup: r.stackup }])
      onHighlight(r.chars)
      setQ('')
    } finally { setBusy(false) }
  }
  const charOf = Object.fromEntries(job.rows.map(r => [r.row_id, r.char_id]))
  return (
    <div className="panel ask">
      <h3>Ask the drawing</h3>
      <p className="fine">Muse Spark answers from the ballooned sheets and the Form 3 lines and names the balloons it used. For a stack-up it only picks the chain; code adds up the limits.</p>
      <div className="chatlog">
        {qs.map((x, k) => (
          <div key={k} className="qa">
            <p className="q">{x.q}</p>
            <p className="a">{x.a}</p>
            {x.rows.length > 0 && <div className="cites">{x.rows.map(r => <button key={r} className="cite" onClick={() => onFocus(charOf[r])}>{r}</button>)}</div>}
            {x.stackup && (
              <table className="stack">
                <thead><tr><th colSpan={3}>{x.stackup.what || 'Stack-up'}</th></tr></thead>
                <tbody>
                  {x.stackup.given.map((g, n) => <tr key={'g' + n}><td>{g.sign > 0 ? '+' : '−'}</td><td>{g.what} (given)</td><td className="lim">{g.value}</td></tr>)}
                  {x.stackup.links.map(l => <tr key={l.row_id} onClick={() => onFocus(charOf[l.row_id])}><td>{l.sign > 0 ? '+' : '−'}</td><td>{l.row_id} · {l.requirement}</td><td className="lim">{l.lower} – {l.upper}</td></tr>)}
                  <tr className="tot"><td /><td>Worst case</td><td className="lim">{x.stackup.worst_case.min} – {x.stackup.worst_case.max}</td></tr>
                  <tr className="tot"><td /><td>Statistical (RSS)</td><td className="lim">{x.stackup.nominal} ± {x.stackup.rss.plus_minus}</td></tr>
                </tbody>
              </table>
            )}
          </div>
        ))}
        {busy && <p className="fine pulse">Muse Spark is reading the drawing…</p>}
      </div>
      {qs.length === 0 && <div className="examples">{EXAMPLES.map(e => <button key={e} className="cite wide" onClick={() => ask(e)}>{e}</button>)}</div>}
      <form className="askbox" onSubmit={e => { e.preventDefault(); ask(q) }}>
        <input value={q} onChange={e => setQ(e.target.value)} placeholder="Ask about this drawing" disabled={busy} />
        <button className="btn primary" disabled={busy || !q.trim()}>Ask</button>
      </form>
      {err && <p className="err">{err}</p>}
    </div>
  )
}

/* ------------------------------------------------------------------ supplier report check */
interface Finding {
  kind: string
  message: string
  supplier_row: { where: string; char_no: string; requirement: string; lower: string; upper: string; result: string; accept: string } | null
  drawing: { char_id: string; number: string; row_id: string; requirement: string; lower: string; upper: string } | null
}

interface Check {
  ok: boolean
  verdict: string
  report: string
  summary: Record<string, number>
  findings: Finding[]
  rows_read: number
  drawing_lines: number
  matched: number
}

const KIND: Record<string, string> = {
  missing_row: 'On the drawing, missing from the report',
  requirement_mismatch: 'Copied wrong',
  accepted_out_of_tolerance: 'Out of tolerance but accepted',
  missing_result: 'No result',
  duplicate_number: 'Number used twice',
  extra_row: 'Not on the drawing',
  number_mismatch: 'Under the wrong number',
}

export function SupplierPanel({ job, onFocus }: { job: Job; onFocus: Focus }) {
  const [samples, setSamples] = useState<string[]>([])
  const [res, setRes] = useState<Check | null>(null)
  const [busy, setBusy] = useState('')
  const [err, setErr] = useState('')
  useEffect(() => { getJSON<string[]>(`/api/jobs/${job.id}/supplier-samples`).then(setSamples).catch(() => {}) }, [job.id])
  const run = async (fd: FormData, label: string) => {
    setBusy(label); setErr(''); setRes(null)
    try {
      const r = await fetch(`/api/jobs/${job.id}/supplier-report`, { method: 'POST', body: fd }).then(r => r.json())
      if (r.error) setErr(r.error); else setRes(r)
    } finally { setBusy('') }
  }
  return (
    <div className="panel">
      <h3>Check a supplier's report</h3>
      <p className="fine">A buyer gets an AS9102 report from a supplier. Whiteout reads it (a spreadsheet in code, a PDF or photo with Muse Spark) and checks every row against this drawing.</p>
      <div className="row wrap">
        {samples.map(s => <button key={s} className="btn" disabled={!!busy} onClick={() => { const fd = new FormData(); fd.append('sample', s); run(fd, s) }}>Try: {s}</button>)}
        <label className="btn">Upload a report…<input type="file" hidden accept=".xlsx,.xlsm,.pdf,.png,.jpg,.jpeg" onChange={e => { const f = e.target.files?.[0]; if (f) { const fd = new FormData(); fd.append('file', f); run(fd, f.name) } }} /></label>
      </div>
      {busy && <p className="fine pulse">Checking {busy}…</p>}
      {err && <p className="err">{err}</p>}
      {res && (
        <div className="check">
          <p className={`verdict ${res.ok ? 'ok' : 'bad'}`}>{res.verdict}</p>
          <p className="fine">{res.rows_read} rows read · {res.matched} matched to the {res.drawing_lines} lines on the drawing</p>
          {res.findings.map((f, k) => (
            <div key={k} className={`finding ${f.kind}`} onClick={() => f.drawing && onFocus(f.drawing.char_id)}>
              <span className="tag amber">{KIND[f.kind] ?? f.kind}</span>
              <p>{f.message}</p>
              {f.supplier_row && <p className="fine">Report {f.supplier_row.where}: “{f.supplier_row.requirement}” {f.supplier_row.lower || f.supplier_row.upper ? `${f.supplier_row.lower} – ${f.supplier_row.upper}` : ''} {f.supplier_row.result ? `· result ${f.supplier_row.result}` : ''} {f.supplier_row.accept ? `· ${f.supplier_row.accept}` : ''}</p>}
              {f.drawing && <p className="fine">Drawing balloon {f.drawing.row_id}: “{f.drawing.requirement}” {f.drawing.lower || f.drawing.upper ? `${f.drawing.lower} – ${f.drawing.upper}` : ''}</p>}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ drawing against the 3D model */
interface MItem { id: string; type: string; text: string; count?: number }
interface DItem { id: string; number: string; text: string; sheet: number }
interface ModelCheck {
  model_file: string
  model_units: string
  drawing_units: string
  summary: Record<string, number>
  matched: { model: MItem; drawing: DItem }[]
  differs: { model: MItem; drawing: DItem; fields: { field: string; model: string; drawing: string }[] }[]
  model_only: MItem[]
  drawing_only: DItem[]
  warnings: string[]
  holes: { id: string; xyz: number[]; diameter: number }[]
}

export function ModelPanel({ job, onFocus }: { job: Job; onFocus: Focus }) {
  const [samples, setSamples] = useState<string[]>([])
  const [res, setRes] = useState<ModelCheck | null>(null)
  const [busy, setBusy] = useState('')
  const [err, setErr] = useState('')
  const [showMatched, setShowMatched] = useState(false)
  useEffect(() => { getJSON<string[]>(`/api/jobs/${job.id}/model-samples`).then(setSamples).catch(() => {}) }, [job.id])
  const run = async (fd: FormData, label: string) => {
    setBusy(label); setErr(''); setRes(null)
    try {
      const r = await fetch(`/api/jobs/${job.id}/model`, { method: 'POST', body: fd }).then(r => r.json())
      if (r.error) setErr(r.error); else setRes(r)
    } finally { setBusy('') }
  }
  return (
    <div className="panel">
      <h3>Drawing against the 3D model</h3>
      <p className="fine">In model-based definition the STEP model carries its own dimensions and tolerances (AP242 PMI), and drawing and model drift apart. Whiteout reads the model's PMI in code and compares it, line by line, with what Muse Spark read on the drawing. Holes found in the model give the CMM program real positions.</p>
      <div className="row wrap">
        {samples.map(s => <button key={s} className="btn" disabled={!!busy} onClick={() => { const fd = new FormData(); fd.append('sample', s); run(fd, s) }}>Try: {s}</button>)}
        <label className="btn">Upload a STEP model…<input type="file" hidden accept=".stp,.step" onChange={e => { const f = e.target.files?.[0]; if (f) { const fd = new FormData(); fd.append('file', f); run(fd, f.name) } }} /></label>
      </div>
      {busy && <p className="fine pulse">Reading {busy}…</p>}
      {err && <p className="err">{err}</p>}
      {res && (
        <div className="check">
          <div className="chips">
            <span className="chip"><b>{res.summary.matched}</b> agree</span>
            <span className={`chip ${res.summary.differs ? 'red' : ''}`}><b>{res.summary.differs}</b> differ</span>
            <span className={`chip ${res.summary.drawing_only ? 'amber' : ''}`}><b>{res.summary.drawing_only}</b> only on the drawing</span>
            <span className={`chip ${res.summary.model_only ? 'amber' : ''}`}><b>{res.summary.model_only}</b> only in the model</span>
            <span className="chip blue"><b>{res.holes.length}</b> holes located for the CMM</span>
          </div>
          <p className="fine">{res.model_file} · model in {res.model_units === 'in' ? 'inches' : 'millimeters'}, drawing in {res.drawing_units === 'in' ? 'inches' : 'millimeters'}</p>
          {res.warnings.map((w, k) => <p key={k} className="verdict bad">{w}</p>)}
          {res.differs.map((d, k) => (
            <div key={'d' + k} className="finding" onClick={() => onFocus(d.drawing.id)}>
              <span className="tag red">Differs</span>
              <p>Balloon {d.drawing.number}: drawing <b>{d.drawing.text}</b>, model <b>{d.model.text}</b></p>
              <p className="fine">{d.fields.map(f => `${f.field}: model ${f.model}, drawing ${f.drawing}`).join(' · ')}</p>
            </div>
          ))}
          {res.drawing_only.map((d, k) => (
            <div key={'o' + k} className="finding" onClick={() => onFocus(d.id)}>
              <span className="tag amber">Only on the drawing</span>
              <p>Balloon {d.number}: <b>{d.text}</b> has no matching PMI in the model.</p>
            </div>
          ))}
          {res.model_only.map((m, k) => (
            <div key={'m' + k} className="finding">
              <span className="tag amber">Only in the model</span>
              <p><b>{m.text}</b> is in the model but not on the drawing.</p>
            </div>
          ))}
          <label className="fine"><input type="checkbox" checked={showMatched} onChange={e => setShowMatched(e.target.checked)} /> show the {res.matched.length} that agree</label>
          {showMatched && <table className="plan"><tbody>{res.matched.map((m, k) => (
            <tr key={k} onClick={() => onFocus(m.drawing.id)}><td className="num">{m.drawing.number}</td><td><span className="rq">{m.drawing.text}</span></td><td><span className="rq">{m.model.text}</span></td></tr>
          ))}</tbody></table>}
        </div>
      )}
    </div>
  )
}
