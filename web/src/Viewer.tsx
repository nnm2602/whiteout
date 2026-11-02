import { useCallback, useEffect, useImperativeHandle, useRef, useState, forwardRef, type ReactNode } from 'react'
import type { Box } from './api'

export interface ViewerHandle {
  focus: (b: Box) => void
  fit: () => void
}

interface Props {
  width: number
  height: number
  layers: { src: string; opacity: number; box?: [number, number, number, number] }[]
  children?: ReactNode // SVG content in sheet pixels
  onSheetClick?: (x: number, y: number) => void
}

/** Pan and zoom a drawing sheet with an SVG overlay drawn in sheet pixels. */
export const Viewer = forwardRef<ViewerHandle, Props>(function Viewer({ width, height, layers, children, onSheetClick }, ref) {
  const host = useRef<HTMLDivElement>(null)
  const [view, setView] = useState({ s: 0.2, x: 0, y: 0 })
  const drag = useRef<{ x: number; y: number; vx: number; vy: number; moved: boolean } | null>(null)
  const anim = useRef<number>(0)

  const fitView = useCallback(() => {
    const el = host.current
    if (!el || !width) return
    const s = Math.min(el.clientWidth / width, el.clientHeight / height) * 0.98
    setView({ s, x: (el.clientWidth - width * s) / 2, y: (el.clientHeight - height * s) / 2 })
  }, [width, height])

  useEffect(() => {
    fitView()
    const ro = new ResizeObserver(fitView)
    if (host.current) ro.observe(host.current)
    return () => ro.disconnect()
  }, [fitView])

  const animateTo = (target: { s: number; x: number; y: number }) => {
    cancelAnimationFrame(anim.current)
    const start = performance.now()
    const from = { ...view }
    const step = (t: number) => {
      const k = Math.min(1, (t - start) / 450)
      const e = 1 - Math.pow(1 - k, 3)
      setView({ s: from.s + (target.s - from.s) * e, x: from.x + (target.x - from.x) * e, y: from.y + (target.y - from.y) * e })
      if (k < 1) anim.current = requestAnimationFrame(step)
    }
    anim.current = requestAnimationFrame(step)
  }

  useImperativeHandle(ref, () => ({
    fit: fitView,
    focus: (b: Box) => {
      const el = host.current
      if (!el) return
      const bw = Math.max(b[2] - b[0], 200), bh = Math.max(b[3] - b[1], 120)
      const s = Math.min(el.clientWidth / (bw * 4), el.clientHeight / (bh * 5), 1.6)
      const cx = (b[0] + b[2]) / 2, cy = (b[1] + b[3]) / 2
      animateTo({ s, x: el.clientWidth / 2 - cx * s, y: el.clientHeight / 2 - cy * s })
    },
  }))

  const onWheel = (e: React.WheelEvent) => {
    const el = host.current
    if (!el) return
    const r = el.getBoundingClientRect()
    const mx = e.clientX - r.left, my = e.clientY - r.top
    const f = Math.exp(-e.deltaY * 0.0015)
    setView(v => {
      const s = Math.min(4, Math.max(0.03, v.s * f))
      return { s, x: mx - ((mx - v.x) * s) / v.s, y: my - ((my - v.y) * s) / v.s }
    })
  }

  return (
    <div
      ref={host}
      className="viewer"
      onWheel={onWheel}
      onPointerDown={e => {
        drag.current = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y, moved: false }
        ;(e.target as HTMLElement).setPointerCapture?.(e.pointerId)
      }}
      onPointerMove={e => {
        const d = drag.current
        if (!d) return
        if (Math.abs(e.clientX - d.x) + Math.abs(e.clientY - d.y) > 4) d.moved = true
        setView(v => ({ ...v, x: d.vx + e.clientX - d.x, y: d.vy + e.clientY - d.y }))
      }}
      onPointerUp={e => {
        const d = drag.current
        drag.current = null
        if (d && !d.moved && onSheetClick && host.current) {
          const r = host.current.getBoundingClientRect()
          onSheetClick((e.clientX - r.left - view.x) / view.s, (e.clientY - r.top - view.y) / view.s)
        }
      }}
    >
      <div className="sheet" style={{ width, height, transform: `translate(${view.x}px, ${view.y}px) scale(${view.s})` }}>
        {layers.map((l, i) => (
          <img key={l.src + i} src={l.src} width={l.box ? l.box[2] - l.box[0] : width} height={l.box ? l.box[3] - l.box[1] : height} draggable={false}
            style={{ opacity: l.opacity, position: 'absolute', left: l.box ? l.box[0] : 0, top: l.box ? l.box[1] : 0, transition: 'opacity 600ms ease' }} />
        ))}
        <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} style={{ position: 'absolute', inset: 0 }}>
          {children}
        </svg>
      </div>
      <div className="viewer-tools">
        <button onClick={fitView} title="Fit the sheet">Fit</button>
        <button onClick={() => setView(v => ({ ...v, s: v.s * 1.4 }))} title="Zoom in">+</button>
        <button onClick={() => setView(v => ({ ...v, s: v.s / 1.4 }))} title="Zoom out">−</button>
      </div>
    </div>
  )
})
