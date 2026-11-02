export type Box = [number, number, number, number]

export interface Requirement {
  type: string
  label: string
  text: string
  nominal: number | null
  lower: number | null
  upper: number | null
  attribute: boolean
  tolerance_source: string
  flags: string[]
}

export interface Char {
  id: string
  number: string
  kind: string
  text: string
  multiplier: number
  box: Box
  raw_box: Box | null
  sheet: number
  zone: string
  confidence: string
  found_in: string
  requirements: Requirement[]
  balloon_xy: [number, number] | null
  flags: string[]
  ignored: boolean
}

export interface AuditFind {
  pass_no: number
  text: string
  box: Box
  verdict: 'new' | 'duplicate' | 'not_requirement' | 'fragment'
  reason: string
  char_id: string | null
}

export interface SheetInfo {
  index: number
  width: number
  height: number
  dpi: number
  part_number: string
  part_name: string
  drawing_number: string
  revision: string
  units: string
  material: string
  finish: string
  general_tolerances: { applies_to: string; plus_minus: string }[]
  zone_columns: string[]
  zone_rows: string[]
  has_text_layer: boolean
}

export interface Sheet {
  info: SheetInfo
  chars: Char[]
  audit: { pass_no: number; finds: AuditFind[]; new_count: number }[]
  ignored: AuditFind[]
  timings: Record<string, number>
  residue_passes: number
}

export interface Row {
  row_id: string
  char_id: string
  number: string
  sheet: number
  zone: string
  requirement: string
  label: string
  type: string
  lower: string
  upper: string
  tolerance_source: string
  attribute: boolean
  instance: string
  found_in: string
  flags: string[]
  measured: string
  status: '' | 'pass' | 'fail' | 'needs review'
}

export interface DeltaSheet {
  unchanged: { new: string; old: string }[]
  changed: { new: string; old: string; old_text: string; new_text: string }[]
  added: string[]
  new_in_report: string[]
  removed: { old: string; text: string; box: Box }[]
  diff_regions: Box[]
}

export interface Job {
  created: number
  id: string
  name: string
  status: 'queued' | 'running' | 'done' | 'error'
  error: string
  model: string
  notes: string[]
  usage: { calls?: number; cached?: number; usd?: number; input_tokens?: number; output_tokens?: number }
  results: Record<string, string>
  revision_of: string
  delta: { base_job?: string; sheets?: DeltaSheet[]; summary?: Record<string, number> }
  sheets: Sheet[]
  rows: Row[]
}

export interface Sample {
  id: string
  name: string
  blurb: string
  thumb: string
  job?: string
}

export async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText)
  return r.json()
}

export async function startJob(form: FormData): Promise<{ id: string }> {
  const r = await fetch('/api/jobs', { method: 'POST', body: form })
  const j = await r.json()
  if (!r.ok) throw new Error(j.error || 'upload failed')
  return j
}

export async function putResults(id: string, results: Record<string, string>): Promise<{ rows: Row[] }> {
  const r = await fetch(`/api/jobs/${id}/results`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ results }),
  })
  return r.json()
}

export const img = (job: string, name: string) => `/api/jobs/${job}/img/${name}.png`
