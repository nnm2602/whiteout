import { useEffect, useMemo, useRef, useState } from 'react'
import { getJSON, img, putResults, startJob, type Box, type Char, type Job, type Row, type Sample } from './api'
import { Viewer, type ViewerHandle } from './Viewer'
import { Recorder } from './voice'
import { AskPanel, MeasurePanel, ModelPanel, PlanPanel, SupplierPanel } from './Panels'

function useRoute() {
  const [hash, setHash] = useState(location.hash)
  useEffect(() => {
    const f = () => setHash(location.hash)
    addEventListener('hashchange', f)
    return () => removeEventListener('hashchange', f)
  }, [])
  const m = hash.match(/^#\/job\/([\w-]+)/)
  return m ? { page: 'job' as const, id: m[1] } : { page: 'home' as const, id: '' }
}

export default function App() {
  const route = useRoute()
  // A saved sample opened by its direct link moves to a private copy, so edits stay with this visitor.
  useEffect(() => {
    if (route.page !== 'job') return
    getJSON<Sample[]>('/api/samples').then(list => {
      const s = list.find(x => x.job === route.id)
      if (!s) return
      const fd = new FormData(); fd.append('sample', s.id)
      return startJob(fd).then(({ id }) => location.replace(`#/job/${id}`))
    }).catch(() => {})
  }, [route.page, route.id])
  return (
    <div className="app">
      <header className="top">
        <a href="#/" className="brand">
          <span className="mark" aria-hidden>
            <svg viewBox="0 0 24 24" width="22" height="22"><circle cx="12" cy="12" r="9.5" fill="none" stroke="currentColor" strokeWidth="2" /><path d="M7 12.5l3 3 7-7" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" /></svg>
          </span>
          Whiteout
        </a>
        <span className="tagline">First article inspection from the drawing to the signed report, on Meta's Muse models.</span>
      </header>
      {route.page === 'home' ? <Home /> : <JobPage id={route.id} key={route.id} />}
    </div>
  )
}

/* ------------------------------------------------------------------ home */
function Home() {
  const [samples, setSamples] = useState<Sample[]>([])
  const [busy, setBusy] = useState('')
  const [err, setErr] = useState('')
  const [engines, setEngines] = useState<{ offline: boolean }>({ offline: false })
  const [engine, setEngine] = useState('spark')
  const [drag, setDrag] = useState(false)

  useEffect(() => {
    getJSON<Sample[]>('/api/samples').then(setSamples).catch(() => {})
    getJSON<{ offline: boolean }>('/api/engines').then(setEngines).catch(() => {})
  }, [])

  const go = async (form: FormData, label: string) => {
    setBusy(label); setErr('')
    try {
      form.append('engine', engine)
      const { id } = await startJob(form)
      location.hash = `#/job/${id}`
    } catch (e) { setErr(String((e as Error).message)) } finally { setBusy('') }
  }
  const upload = (f: File | undefined) => {
    if (!f) return
    const fd = new FormData(); fd.append('file', f); go(fd, f.name)
  }

  return (
    <main className="home">
      <section className="hero">
        <h1>First article inspection starts with a ballooned drawing.<br /><em>Whiteout makes sure nothing is left off it.</em></h1>
        <p className="lede">
          Drop in a drawing: a CAD PDF, an old scan, or a photo of a print. Muse Spark reads every dimension, tolerance,
          GD&amp;T frame and note, and Whiteout numbers them and fills AS9102 Forms 1 to 3. Then it erases everything
          it found and has Muse Spark search the blank page for anything left. From there it plans the measuring, talks
          the inspector through it hands-free, writes a starting CMM program, answers questions about the drawing,
          and checks the reports suppliers send back.
        </p>
        <div
          className={`drop ${drag ? 'over' : ''}`}
          onDragOver={e => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)}
          onDrop={e => { e.preventDefault(); setDrag(false); upload(e.dataTransfer.files[0]) }}
        >
          <input id="file" type="file" accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff,.webp" onChange={e => upload(e.target.files?.[0])} hidden />
          <label htmlFor="file" className="btn primary big">{busy ? `Starting ${busy}…` : 'Choose a drawing'}</label>
          <span className="hint">or drop a PDF, PNG, JPEG or TIFF here (up to 6 sheets, 25 MB)</span>
          <div className="engine">
            <label><input type="radio" checked={engine === 'spark'} onChange={() => setEngine('spark')} /> Muse Spark 1.3 (Meta Model API)</label>
            <label className={engines.offline ? '' : 'off'} title={engines.offline ? '' : 'The offline server is not running right now'}>
              <input type="radio" disabled={!engines.offline} checked={engine === 'offline'} onChange={() => setEngine('offline')} /> Muse Glimmer open weights on a rented GPU (no model API)
            </label>
          </div>
          {err && <p className="err">{err}</p>}
        </div>
      </section>

      {samples.length > 0 && (
        <section className="samples">
          <h2>Or open a drawing we already ran</h2>
          <div className="cards">
            {samples.map(s => (
              <button key={s.id} className="card" onClick={() => { const fd = new FormData(); fd.append('sample', s.id); go(fd, s.name) }}>
                <img src={`/api/samples/${s.id}/thumb.png`} alt="" />
                <strong>{s.name}</strong>
                <span>{s.blurb}</span>
              </button>
            ))}
          </div>
        </section>
      )}

      <section className="how">
        {[
          ['Read', 'The sheet is cut into tiles small enough for Muse Spark to read at full resolution. Every requirement comes back with its text, its tolerance split into Form 3 lines, and its place on the sheet.'],
          ['Check in code', 'Boxes are snapped onto the real ink. Limits are recomputed from the printed text, and any number the model gives that is not on the drawing is flagged.'],
          ['Whiteout', 'Everything found is painted white. Muse Spark searches what is left, on tiles shifted half a step, until a pass finds nothing new. Misses show up red on a blank sheet.'],
          ['Report', 'Balloons in reading order with zones, AS9102 Forms 1 to 3 as a spreadsheet, results typed, imported from a CMM file or spoken, and a new revision carries unchanged results forward.'],
          ['Plan', 'An instrument and a method for every line, chosen in code from the tolerance band, grouped into setups, plus a starting CMM program in DMIS.'],
          ['Measure', 'Hands-free: each line is read aloud, the inspector says the value (Muse Voice Transcribe) or photographs the gauge (Muse Spark), and pass or fail is decided in code.'],
          ['Ask', 'Questions about the drawing answered from the ballooned sheets, with the balloons cited. Tolerance stack-ups are added up in code, not by the model.'],
          ['Check suppliers', 'A supplier\'s AS9102 report, as a spreadsheet or a scanned PDF, checked row by row against the drawing: missing lines, copied limits, accepted failures.'],
        ].map(([t, d], i) => (
          <div key={t} className="step"><span className="n">{i + 1}</span><h3>{t}</h3><p>{d}</p></div>
        ))}
      </section>
      <footer className="foot">
        Built during the Meta Global AI Developer Hackathon (Oct 27 to Nov 4 2026) with Muse Spark, Muse Voice Transcribe and Muse Glimmer.
        A person signs the report; Whiteout prepares it.
      </footer>
    </main>
  )
}

/* ------------------------------------------------------------------ job */
type Mode = 'balloons' | 'whiteout' | 'changes' | 'redraw'
type Panel = 'form3' | 'plan' | 'measure' | 'ask' | 'supplier' | 'model'
const PANELS: [Panel, string][] = [['form3', 'Form 3'], ['plan', 'Plan'], ['measure', 'Measure'], ['ask', 'Ask'], ['supplier', 'Supplier'], ['model', '3D model']]

interface Live {
  stage: string
  tiles: Box[]
  read: Box[]
  finds: Box[]
  pass: number
  log: string[]
  done: number
}

function Progress({ job, live }: { job: Job; live: Live }) {
  const [, tick] = useState(0)
  useEffect(() => {
    const t = setInterval(() => tick(n => n + 1), 1000)
    return () => clearInterval(t)
  }, [])
  const secs = Math.max(0, Math.round(Date.now() / 1000 - job.created))
  const total = live.tiles.length
  const reading = live.stage === 'read' && total > 0
  const step = live.stage === 'audit' ? 3 : live.stage === 'read' ? 2 : 1
  const label = live.stage === 'audit' ? `Painting out what was found and searching what is left, pass ${live.pass}`
    : reading ? `Reading the drawing, ${Math.min(live.done, total)} of ${total} sections done`
    : live.stage === 'read' ? 'Reading the drawing'
    : live.stage === 'overview' ? 'Finding the title block and zones'
    : 'Preparing the drawing'
  return (
    <div className="progress" role="status">
      <div className="progress-top">
        <span className="spinner" aria-hidden />
        <strong>Step {step} of 3: {label}</strong>
        <span className="progress-time">{Math.floor(secs / 60)}:{String(secs % 60).padStart(2, '0')}</span>
      </div>
      <div className="progress-track">
        <div className={reading ? 'progress-fill' : 'progress-fill moving'} style={reading ? { width: `${Math.min(100, (live.done / total) * 100)}%` } : undefined} />
      </div>
      <p>This usually takes about 2 minutes. Keep this page open; requirements appear on the drawing as they are found.</p>
    </div>
  )
}

function JobPage({ id }: { id: string }) {
  const [job, setJob] = useState<Job | null>(null)
  const [err, setErr] = useState('')
  const [sheet, setSheet] = useState(0)
  const [mode, setMode] = useState<Mode>('balloons')
  const [sel, setSel] = useState<string>('')
  const [live, setLive] = useState<Live>({ stage: '', tiles: [], read: [], finds: [], pass: 0, log: [], done: 0 })
  const [residueTick, setResidueTick] = useState(0)
  const [panel, setPanel] = useState<Panel>('form3')
  const [hl, setHl] = useState<string[]>([])
  const viewer = useRef<ViewerHandle>(null)

  const refresh = () => getJSON<Job>(`/api/jobs/${id}`).then(setJob).catch(e => setErr(String(e.message)))

  useEffect(() => {
    refresh()
    const es = new EventSource(`/api/jobs/${id}/events`)
    es.onmessage = m => {
      const ev = JSON.parse(m.data)
      if (ev.type === 'end') { es.close(); refresh(); return }
      // sheets run in parallel; the live picture follows sheet 1, the log shows every sheet
      const other = ev.sheet !== undefined && ev.sheet !== 0
      if (other && ev.type !== 'stage' && ev.type !== 'read_done' && ev.type !== 'audit_pass' && ev.type !== 'error') return
      setLive(l => {
        const n = { ...l }
        if (ev.type === 'stage' && other) {
          n.log = [...l.log, ev.stage === 'audit' ? `Sheet ${ev.sheet + 1}: searching the erased page, pass ${ev.pass}` : ev.stage === 'read' ? `Reading ${ev.tiles.length} tiles of sheet ${ev.sheet + 1}` : `Finding the title block and zones of sheet ${ev.sheet + 1}`]
          return n
        }
        if (ev.type === 'stage') {
          n.stage = ev.stage
          if (ev.stage === 'read') { n.tiles = ev.tiles; n.read = []; n.finds = []; n.done = 0 }
          if (ev.stage === 'audit') n.pass = ev.pass
          n.log = [...l.log, ev.stage === 'audit' ? `Searching the erased page, pass ${ev.pass}` : ev.stage === 'read' ? `Reading ${ev.tiles.length} tiles of sheet ${ev.sheet + 1}` : `Finding the title block and zones of sheet ${ev.sheet + 1}`]
        }
        if (ev.type === 'tile' && ev.stage === 'read') { n.read = [...l.read, ...ev.items]; n.done = l.done + 1 }
        if (ev.type === 'tile' && ev.stage !== 'read') n.finds = [...l.finds, ...ev.items.filter((_: Box, i: number) => (ev.whats?.[i] ?? 'requirement') === 'requirement')]
        if (ev.type === 'read_done') n.log = [...n.log, `${ev.kept} requirements after merging ${ev.raw} tile reads`]
        if (ev.type === 'audit_pass') n.log = [...n.log, ev.new ? `Pass ${ev.pass}: ${ev.new} missed requirement${ev.new > 1 ? 's' : ''} found` : `Pass ${ev.pass}: nothing left`]
        if (ev.type === 'error') n.log = [...n.log, `Error: ${ev.error}`]
        return n
      })
      if (ev.type === 'residue' && !other) setResidueTick(t => t + 1)
      if (ev.type === 'done') refresh()
    }
    return () => es.close()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id])

  if (err) return <main className="center"><p className="err">{err}</p><a href="#/">Back</a></main>
  if (!job) return <main className="center"><p>Loading…</p></main>

  const running = job.status === 'running' || job.status === 'queued'
  const s = job.sheets[sheet]
  const focusChar = (cid: string) => {
    const ch = job.sheets.flatMap(x => x.chars).find(x => x.id === cid)
    if (!ch) return
    setSel(cid)
    if (ch.sheet !== sheet) setSheet(ch.sheet)
    setMode('balloons')
    setTimeout(() => viewer.current?.focus(ch.box), 60)
  }
  const onRows = (rows: Row[]) => setJob(j => (j ? { ...j, rows } : j))
  const size = s ? { w: s.info.width, h: s.info.height } : null
  return (
    <main className="job">
      <section className="left">
        <div className="bar">
          <div className="tabs">
            <button className={mode === 'balloons' ? 'on' : ''} onClick={() => setMode('balloons')}>Ballooned</button>
            <button className={mode === 'whiteout' ? 'on' : ''} onClick={() => setMode('whiteout')}>Whiteout check</button>
            {job.revision_of && <button className={mode === 'changes' ? 'on' : ''} onClick={() => setMode('changes')}>Changes</button>}
            {s && !s.info.has_text_layer && job.status === 'done' && <button className={mode === 'redraw' ? 'on' : ''} onClick={() => setMode('redraw')}>Redrawn</button>}
          </div>
          {job.sheets.length > 1 && (
            <div className="tabs small">
              {job.sheets.map((_, i) => <button key={i} className={i === sheet ? 'on' : ''} onClick={() => setSheet(i)}>Sheet {i + 1}</button>)}
            </div>
          )}
        </div>
        {running && !s ? (
          <LiveSheet id={id} sheet={sheet} live={live} tick={residueTick} />
        ) : s && size ? (
          mode === 'whiteout' ? (
            <WhiteoutView key={`${sheet}-${residueTick}`} job={job} sheet={sheet} />
          ) : mode === 'redraw' ? (
            <RedrawView key={sheet} job={job} sheet={sheet} />
          ) : (
            <Viewer ref={viewer} width={size.w} height={size.h}
              layers={[{ src: img(id, `sheet${sheet}`), opacity: 1 }]}>
              {mode === 'balloons' ? <Balloons chars={s.chars} sel={sel} hl={hl} onSel={setSel} r={Math.max(14, Math.round(0.11 * s.info.dpi))} />
                : <Changes job={job} sheet={sheet} />}
            </Viewer>
          )
        ) : null}
        {mode === 'changes' && job.delta.summary && (
          <div className="wo-panel legend">
            <span><i className="sw changed" /> changed {job.delta.summary.changed}</span>
            <span><i className="sw added" /> added {job.delta.summary.added}</span>
            <span><i className="sw removed" /> removed {job.delta.summary.removed}</span>
            <span><i className="sw newread" /> not in the earlier report {job.delta.summary.new_in_report ?? 0}</span>
            <span><i className="sw same" /> unchanged {job.delta.summary.unchanged}, {job.delta.summary.carried_values} results carried forward</span>
            <span><i className="sw diffreg" /> ink that differs between revisions</span>
          </div>
        )}
        {running && s && <div className="livebar">{live.log[live.log.length - 1] || 'Working…'}</div>}
      </section>
      <section className="right">
        {running && <Progress job={job} live={live} />}
        <Summary job={job} sheet={sheet} running={running} live={live} />
        {job.status === 'error' && <p className="err">{job.error}</p>}
        {job.status === 'done' && (
          <div className="tabs panels">
            {PANELS.map(([k, label]) => <button key={k} className={panel === k ? 'on' : ''} onClick={() => { setPanel(k); setHl([]) }}>{label}</button>)}
          </div>
        )}
        {(panel === 'form3' || job.status !== 'done') && <>
          {job.status === 'done' && <Actions job={job} onRows={onRows} />}
          <Form3 job={job} sel={sel} onSel={c => focusChar(c)} onRows={onRows} onChanged={refresh} />
        </>}
        {job.status === 'done' && panel === 'plan' && <PlanPanel job={job} onFocus={focusChar} onMeasure={() => setPanel('measure')} />}
        {job.status === 'done' && panel === 'measure' && <MeasurePanel job={job} onFocus={focusChar} onRows={onRows} />}
        {job.status === 'done' && panel === 'ask' && <AskPanel job={job} onFocus={focusChar} onHighlight={setHl} />}
        {job.status === 'done' && panel === 'supplier' && <SupplierPanel job={job} onFocus={focusChar} />}
        {job.status === 'done' && panel === 'model' && <ModelPanel job={job} onFocus={focusChar} />}
      </section>
    </main>
  )
}

function LiveSheet({ id, sheet, live, tick }: { id: string; sheet: number; live: Live; tick: number }) {
  const [size, setSize] = useState<{ w: number; h: number } | null>(null)
  useEffect(() => {
    // the sheet image appears a moment after the upload is converted; keep trying until it does
    let stop = false
    let n = 0
    const tryLoad = () => {
      const im = new Image()
      im.onload = () => { if (!stop) setSize({ w: im.naturalWidth, h: im.naturalHeight }) }
      im.onerror = () => { if (!stop && n++ < 120) setTimeout(tryLoad, 1000) }
      im.src = img(id, `sheet${sheet}`) + `?try=${n}`
    }
    tryLoad()
    return () => { stop = true }
  }, [id, sheet])
  if (!size) return <div className="viewer center"><p className="pulse">Loading the drawing…</p></div>
  const residue = tick > 0 ? img(id, `residue${sheet}_${Math.max(1, live.pass)}`) + `?t=${tick}` : ''
  return (
    <Viewer width={size.w} height={size.h} layers={[{ src: img(id, `sheet${sheet}`), opacity: 1 }, ...(residue ? [{ src: residue, opacity: live.stage === 'audit' ? 1 : 0 }] : [])]}>
      {live.stage === 'read' && live.tiles.map((t, i) => (
        <rect key={i} x={t[0]} y={t[1]} width={t[2] - t[0]} height={t[3] - t[1]} className="tile" />
      ))}
      {live.stage !== 'audit' && live.read.map((b, i) => (
        <rect key={i} x={b[0] - 4} y={b[1] - 4} width={b[2] - b[0] + 8} height={b[3] - b[1] + 8} className="found pop" />
      ))}
      {live.stage === 'audit' && live.finds.map((b, i) => (
        <rect key={i} x={b[0] - 6} y={b[1] - 6} width={b[2] - b[0] + 12} height={b[3] - b[1] + 12} className="checking" />
      ))}
    </Viewer>
  )
}

function Balloons({ chars, sel, hl, onSel, r }: { chars: Char[]; sel: string; hl: string[]; onSel: (id: string) => void; r: number }) {
  return (
    <g>
      {chars.filter(c => !c.ignored && c.balloon_xy).map(c => {
        const [bx, by] = c.balloon_xy!
        const tx = Math.min(Math.max(bx, c.box[0]), c.box[2]), ty = Math.min(Math.max(by, c.box[1]), c.box[3])
        const cls = `bal ${c.found_in.startsWith('audit') ? 'audit' : c.found_in === 'manual' ? 'manual' : ''} ${c.flags.length ? 'flag' : ''} ${sel === c.id ? 'sel' : ''} ${hl.includes(c.id) ? 'hl' : ''}`
        return (
          <g key={c.id} className={cls} onPointerUp={e => { e.stopPropagation(); onSel(c.id) }}>
            <rect x={c.box[0] - 3} y={c.box[1] - 3} width={c.box[2] - c.box[0] + 6} height={c.box[3] - c.box[1] + 6} className="bx" />
            <line x1={bx} y1={by} x2={tx} y2={ty} />
            <circle cx={bx} cy={by} r={r} />
            <text x={bx} y={by} dy="0.36em" fontSize={r * (c.number.length > 2 ? 0.8 : 1)}>{c.number}</text>
          </g>
        )
      })}
    </g>
  )
}

function WhiteoutView({ job, sheet }: { job: Job; sheet: number }) {
  const s = job.sheets[sheet]
  const passes = s.audit.length
  const [step, setStep] = useState(-1) // -1 original, 0..n erasing, 'done'
  const [phase, setPhase] = useState<'idle' | 'erasing' | 'residue'>('residue')
  const chars = useMemo(() => [...s.chars].filter(c => !c.ignored && c.found_in === 'read')
    .sort((a, b) => Number(a.number) - Number(b.number)), [s])
  const [showIgnored, setShowIgnored] = useState(false)
  const finds = s.audit.flatMap(a => a.finds)

  const replay = () => {
    setPhase('erasing'); setStep(0)
    let i = 0
    const t = setInterval(() => {
      i += Math.max(1, Math.round(chars.length / 40))
      setStep(i)
      if (i >= chars.length) { clearInterval(t); setTimeout(() => setPhase('residue'), 400) }
    }, 45)
  }

  const residueSrc = img(job.id, `residue${sheet}_final`)
  const firstResidue = img(job.id, `residue${sheet}_1`)
  return (
    <div className="whiteout">
      <Viewer width={s.info.width} height={s.info.height} layers={[
        { src: img(job.id, `sheet${sheet}`), opacity: phase === 'residue' ? 0 : 1 },
        { src: phase === 'residue' ? residueSrc : firstResidue, opacity: phase === 'residue' ? 1 : 0 },
      ]}>
        {phase === 'erasing' && chars.slice(0, Math.max(0, step)).map(c => (
          <rect key={c.id} x={c.box[0] - 4} y={c.box[1] - 4} width={c.box[2] - c.box[0] + 8} height={c.box[3] - c.box[1] + 8} className="wipe" />
        ))}
        {phase === 'residue' && finds.map((f, i) => f.verdict === 'new' ? (
          <g key={i} className="missed">
            <rect x={f.box[0] - 10} y={f.box[1] - 10} width={f.box[2] - f.box[0] + 20} height={f.box[3] - f.box[1] + 20} />
            <text x={f.box[0] - 10} y={f.box[1] - 18}>{`missed on the first read: ${f.text}`}</text>
          </g>
        ) : showIgnored && (f.verdict === 'not_requirement' || f.verdict === 'fragment') ? (
          <g key={i} className="ignored">
            <rect x={f.box[0] - 4} y={f.box[1] - 4} width={f.box[2] - f.box[0] + 8} height={f.box[3] - f.box[1] + 8} />
          </g>
        ) : null)}
      </Viewer>
      <div className="wo-panel">
        <button className="btn" onClick={replay}>▶ Replay the erase</button>
        <span>{passes} search pass{passes === 1 ? '' : 'es'} on the erased page · {s.audit.reduce((a, p) => a + p.new_count, 0)} missed requirement(s) found · last pass found nothing new: {s.audit.length && s.audit[s.audit.length - 1].new_count === 0 ? 'yes' : 'no'}</span>
        <label><input type="checkbox" checked={showIgnored} onChange={e => setShowIgnored(e.target.checked)} /> show text checked and ruled out ({finds.filter(f => f.verdict === 'not_requirement' || f.verdict === 'fragment').length})</label>
      </div>
    </div>
  )
}

interface RedrawInfo {
  region: [number, number, number, number]
  faithfulness: { line_ink_covered?: number }
  placed: number
  differences: { box: [number, number, number, number]; redraw: string; scan: string; kind: string }[]
}

function RedrawView({ job, sheet }: { job: Job; sheet: number }) {
  const s = job.sheets[sheet]
  const [info, setInfo] = useState<RedrawInfo | null>(null)
  const [state, setState] = useState<'loading' | 'none' | 'working' | 'ready' | 'error'>('loading')
  const [err, setErr] = useState('')
  const [show, setShow] = useState<'redraw' | 'scan' | 'blink'>('redraw')
  const [tick, setTick] = useState(0)
  const [marks, setMarks] = useState(true)
  useEffect(() => {
    fetch(`/api/jobs/${job.id}/redraw/${sheet}`).then(async r => {
      if (r.ok) { setInfo(await r.json()); setState('ready') } else setState('none')
    }).catch(() => setState('none'))
  }, [job.id, sheet])
  useEffect(() => {
    if (show !== 'blink') return
    const t = setInterval(() => setTick(k => k + 1), 900)
    return () => clearInterval(t)
  }, [show])
  const make = async () => {
    setState('working'); setErr('')
    const r = await fetch(`/api/jobs/${job.id}/redraw/${sheet}`, { method: 'POST' })
    const j = await r.json()
    if (r.ok) { setInfo(j); setState('ready') } else { setErr(j.error || 'redraw failed'); setState('error') }
  }
  if (state !== 'ready' || !info) {
    return (
      <div className="viewer center">
        <div className="redraw-empty">
          <p>Redraw this scanned sheet as a clean vector drawing: the line work traced in code, every text read by Muse Spark and set in type, then proofread against the scan.</p>
          {state === 'loading' ? <p className="fine">Loading…</p> : state === 'working'
            ? <p className="pulse">Redrawing and proofreading, about two minutes…</p>
            : <button className="btn primary" onClick={make}>Redraw sheet {sheet + 1}</button>}
          {err && <p className="err">{err}</p>}
        </div>
      </div>
    )
  }
  const redrawOn = show === 'redraw' || (show === 'blink' && tick % 2 === 0)
  const base = `/api/jobs/${job.id}/redraw/${sheet}`
  return (
    <div className="whiteout">
      <Viewer width={s.info.width} height={s.info.height} layers={[
        { src: img(job.id, `sheet${sheet}`), opacity: 1 },
        { src: `${base}/redraw.png`, opacity: redrawOn ? 1 : 0, box: info.region },
      ]}>
        {marks && info.differences.map((d, i) => (
          <g key={i} className="proofmark">
            <rect x={d.box[0] - 8} y={d.box[1] - 8} width={d.box[2] - d.box[0] + 16} height={d.box[3] - d.box[1] + 16} />
            <title>{`${d.kind}: redraw says "${d.redraw}", scan says "${d.scan}"`}</title>
          </g>
        ))}
      </Viewer>
      <div className="wo-panel">
        <div className="tabs small">
          {(['redraw', 'scan', 'blink'] as const).map(k => <button key={k} className={show === k ? 'on' : ''} onClick={() => setShow(k)}>{k === 'redraw' ? 'Redrawn' : k === 'scan' ? 'Scan' : 'Blink'}</button>)}
        </div>
        <label><input type="checkbox" checked={marks} onChange={e => setMarks(e.target.checked)} /> {info.differences.length} places where proofreading found the text differs from the scan</label>
        <a className="btn" href={`${base}/redraw.pdf`}>PDF</a>
        <a className="btn" href={`${base}/redraw.svg`}>SVG</a>
        <span className="fine">Redrawn for reading, not a released drawing. {Math.round((info.faithfulness.line_ink_covered ?? 0) * 100)}% of the scan's line work traced. Check every red mark against the scan.</span>
      </div>
    </div>
  )
}

function Changes({ job, sheet }: { job: Job; sheet: number }) {
  const d = job.delta.sheets?.[sheet]
  if (!d) return null
  const byId = Object.fromEntries(job.sheets[sheet].chars.map(c => [c.id, c]))
  const R = (b: Box, cls: string, k: string) => <rect key={k} x={b[0] - 6} y={b[1] - 6} width={b[2] - b[0] + 12} height={b[3] - b[1] + 12} className={cls} />
  return (
    <g>
      {d.diff_regions.map((b, i) => R(b, 'diffreg', 'd' + i))}
      {d.unchanged.map(u => byId[u.new] && R(byId[u.new].box, 'same', u.new))}
      {d.changed.map(u => byId[u.new] && R(byId[u.new].box, 'changed', u.new))}
      {d.added.map(id => byId[id] && R(byId[id].box, 'added', id))}
      {(d.new_in_report ?? []).map(id => byId[id] && R(byId[id].box, 'newread', id))}
      {d.removed.map(r => R(r.box, 'removed', r.old))}
    </g>
  )
}

function latestRev(job: Job): string {
  const revs = job.sheets.map(s => s.info.revision).filter(Boolean) as string[]
  const letters = revs.map(r => r.trim().toUpperCase()).filter(r => /^[A-Z]+$/.test(r))
  letters.sort((a, b) => a.length - b.length || (a < b ? -1 : a > b ? 1 : 0))
  return letters[letters.length - 1] ?? revs[0] ?? ''
}

function Summary({ job, sheet, running, live }: { job: Job; sheet: number; running: boolean; live: Live }) {
  const s = job.sheets[sheet]
  const all = job.sheets.flatMap(x => x.chars).filter(c => !c.ignored)
  const audit = all.filter(c => c.found_in.startsWith('audit')).length
  const flagged = job.rows.filter(r => r.flags.length).length
  const sec = job.sheets.reduce((a, x) => a + (x.timings.total || 0), 0)
  return (
    <div className="summary">
      <h2 title={job.name}>{job.name}</h2>
      {s && <p className="meta">{[s.info.part_number && `Part ${s.info.part_number}`, latestRev(job) && `Rev ${latestRev(job)}`, s.info.units === 'mm' ? 'millimeters' : 'inches', s.info.material].filter(Boolean).join(' · ')}</p>}
      <div className="chips">
        <span className="chip"><b>{all.length}</b> requirements</span>
        <span className="chip"><b>{job.rows.length}</b> Form 3 lines</span>
        <span className={`chip ${audit ? 'red' : ''}`}><b>{audit}</b> found on the erased page</span>
        {flagged > 0 && <span className="chip amber"><b>{flagged}</b> to check</span>}
        {job.delta.summary && <span className="chip blue"><b>{job.delta.summary.to_measure}</b> to measure in this revision</span>}
      </div>
      <p className="fine">
        {job.model || 'Muse Spark 1.3'} · {job.usage.calls ?? 0} model calls{job.usage.cached ? ` (+${job.usage.cached} cached)` : ''} · ${(job.usage.usd ?? 0).toFixed(2)}{sec ? ` · ${Math.round(sec)} s` : ''}
      </p>
      {running && (
        <ol className="log">{live.log.map((l, i) => <li key={i} className={i === live.log.length - 1 ? 'now' : ''}>{l}</li>)}</ol>
      )}
      {job.notes.length > 0 && <p className="fine">{job.notes.join(' · ')}</p>}
    </div>
  )
}

function Actions({ job, onRows }: { job: Job; onRows: (r: Row[]) => void }) {
  const [msg, setMsg] = useState('')
  const [rec, setRec] = useState<Recorder | null>(null)
  const cmm = async (f?: File) => {
    if (!f) return
    const fd = new FormData(); fd.append('file', f)
    const r = await fetch(`/api/jobs/${job.id}/cmm`, { method: 'POST', body: fd }).then(r => r.json())
    if (r.rows) { onRows(r.rows); setMsg(`Imported ${r.imported} measured values from ${f.name}`) } else setMsg(r.error || 'import failed')
  }
  const revision = async (f?: File) => {
    if (!f) return
    const fd = new FormData(); fd.append('file', f); fd.append('revision_of', job.id)
    try { const { id } = await startJob(fd); location.hash = `#/job/${id}` } catch (e) { setMsg((e as Error).message) }
  }
  const talk = async (down: boolean) => {
    if (down) {
      try { const r = new Recorder(); await r.start(); setRec(r); setMsg('Listening… say "balloon 14, point two five one two"') }
      catch { setMsg('Microphone not available') }
    } else if (rec) {
      const wav = await rec.stop(); setRec(null); setMsg('Transcribing with Muse Voice Transcribe…')
      const fd = new FormData(); fd.append('audio', wav, 'measure.wav')
      const r = await fetch(`/api/jobs/${job.id}/voice`, { method: 'POST', body: fd }).then(r => r.json())
      if (r.rows) { onRows(r.rows); setMsg(`Heard “${r.transcript}” → ${r.entries.map((e: string[]) => `${e[0]}: ${e[1]}`).join(', ') || 'no balloon and value pair'}`) }
      else setMsg(r.error || 'voice failed')
    }
  }
  return (
    <div className="actions">
      <a className="btn primary" href={`/api/jobs/${job.id}/forms.xlsx`}>AS9102 Forms 1–3 (.xlsx)</a>
      <a className="btn" href={`/api/jobs/${job.id}/ballooned.pdf`}>Ballooned drawing (.pdf)</a>
      <label className="btn">Import CMM results (.csv)<input type="file" accept=".csv,.txt" hidden onChange={e => cmm(e.target.files?.[0])} /></label>
      <button className={`btn ${rec ? 'rec' : ''}`} onPointerDown={() => talk(true)} onPointerUp={() => talk(false)} onPointerLeave={() => rec && talk(false)}>
        {rec ? '● Release to send' : '🎙 Hold to dictate results'}
      </button>
      <label className="btn">Next revision arrived…<input type="file" accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff" hidden onChange={e => revision(e.target.files?.[0])} /></label>
      {msg && <p className="fine">{msg}</p>}
    </div>
  )
}

function Form3({ job, sel, onSel, onRows, onChanged }: { job: Job; sel: string; onSel: (charId: string, row: Row) => void; onRows: (r: Row[]) => void; onChanged: () => void }) {
  const [filter, setFilter] = useState<'all' | 'check' | 'audit' | 'fail'>('all')
  const timer = useRef<number>(0)
  const pending = useRef<Record<string, string>>({})
  const [removed, setRemoved] = useState<{ id: string; label: string } | null>(null)
  const setIgnored = async (id: string, ignored: boolean) => {
    const r = await fetch(`/api/jobs/${job.id}/chars/${id}`, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ignored }),
    }).then(x => x.json())
    if (r.rows) onRows(r.rows)
    onChanged()
  }
  const delta = job.delta.sheets
  const changedIds = new Set(delta?.flatMap(d => [...d.changed.map(c => c.new), ...d.added, ...(d.new_in_report ?? [])]) ?? [])
  const rows = job.rows.filter(r => filter === 'all' || (filter === 'check' && r.flags.length) || (filter === 'audit' && r.found_in.startsWith('audit')) || (filter === 'fail' && r.status === 'fail'))
  const save = (rid: string, v: string) => {
    pending.current[rid] = v
    clearTimeout(timer.current)
    timer.current = window.setTimeout(async () => {
      const p = pending.current; pending.current = {}
      const r = await putResults(job.id, p); onRows(r.rows)
    }, 500)
  }
  return (
    <div className="form3">
      <div className="f3head">
        <h3>Form 3 · Characteristic accountability</h3>
        {removed && <span className="removed">Removed {removed.label} <button onClick={() => { setIgnored(removed.id, false); setRemoved(null) }}>Undo</button></span>}
        <select value={filter} onChange={e => setFilter(e.target.value as typeof filter)}>
          <option value="all">All lines</option><option value="check">To check</option>
          <option value="audit">Found on the erased page</option><option value="fail">Out of tolerance</option>
        </select>
      </div>
      <div className="tablewrap">
        <table>
          <thead><tr><th>No.</th><th>Zone</th><th>Requirement</th><th>Limits</th><th>Result</th><th></th><th></th></tr></thead>
          <tbody>
            {rows.map(r => (
              <tr key={r.row_id} className={`${sel === r.char_id ? 'sel' : ''} ${r.found_in.startsWith('audit') ? 'audit' : ''} ${changedIds.has(r.char_id) ? 'changed' : ''}`}
                onClick={() => onSel(r.char_id, r)}>
                <td className="num">{r.row_id}</td>
                <td>{r.zone}</td>
                <td className="req">
                  <span className="rq">{r.requirement}</span>
                  {r.label && <span className="lbl">{r.label}</span>}
                  {r.instance && <span className="lbl">{r.instance}</span>}
                  {r.found_in.startsWith('audit') && <span className="tag red">erased-page search</span>}
                  {r.flags.length > 0 && <span className="tag amber" title={r.flags.join('\n')}>check</span>}
                </td>
                <td className="lim">{r.attribute ? 'attribute' : r.lower || r.upper ? `${r.lower} – ${r.upper}` : r.type === 'basic' ? 'basic' : r.type === 'reference' ? 'ref' : ''}
                  {r.tolerance_source === 'general' && <span className="lbl">title block</span>}</td>
                <td><input defaultValue={r.measured} placeholder={r.attribute ? 'OK / NC' : ''} onClick={e => e.stopPropagation()} onChange={e => save(r.row_id, e.target.value)} /></td>
                <td>{r.status && <span className={`st ${r.status.replace(' ', '-')}`}>{r.status === 'needs review' ? '?' : r.status}</span>}</td>
                <td><button className="rm" title="Remove this line" aria-label={`Remove line ${r.row_id}`} onClick={e => { e.stopPropagation(); setIgnored(r.char_id, true); setRemoved({ id: r.char_id, label: `line ${r.number}` }) }}>×</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
