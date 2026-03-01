/** Vertical colorbar overlay for the 2D map view. */

import { useEffect, useMemo, useRef } from 'react'
import { useColorScaleStore } from '../../stores/colorScaleStore'
import { COLORMAPS } from '../../colormap'

const BAR_WIDTH = 20
const BAR_HEIGHT = 200
const TICK_COUNT = 6

function formatTick(v: number) {
  if (Math.abs(v) >= 100) return v.toFixed(0)
  if (Math.abs(v) >= 1) return v.toFixed(1)
  return v.toFixed(2)
}

export function MapColorbar() {
  const colormap = useColorScaleStore((s) => s.colormap)
  const rangeMin = useColorScaleStore((s) => s.rangeMin)
  const rangeMax = useColorScaleStore((s) => s.rangeMax)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const lut = COLORMAPS[colormap]
    const dpr = window.devicePixelRatio || 1
    canvas.width = BAR_WIDTH * dpr
    canvas.height = BAR_HEIGHT * dpr
    ctx.scale(dpr, dpr)

    // Draw gradient from top (max) to bottom (min)
    for (let y = 0; y < BAR_HEIGHT; y++) {
      const t = 1 - y / (BAR_HEIGHT - 1) // top = max, bottom = min
      const idx = Math.round(t * 255)
      const [r, g, b] = lut[idx]
      ctx.fillStyle = `rgb(${r},${g},${b})`
      ctx.fillRect(0, y, BAR_WIDTH, 1)
    }
  }, [colormap])

  // Generate tick values
  const ticks = useMemo(() => {
    const result: { value: number; y: number }[] = []
    const span = rangeMax - rangeMin
    for (let i = 0; i < TICK_COUNT; i++) {
      const frac = i / (TICK_COUNT - 1)
      const value = rangeMax - frac * span // top = max
      const y = frac * BAR_HEIGHT
      result.push({ value, y })
    }
    return result
  }, [rangeMin, rangeMax])

  return (
    <div className="absolute bottom-4 right-4 z-10 glass-panel rounded-lg p-2 shadow-overlay">
      <div className="text-[10px] font-mono text-zinc-600 mb-1.5 text-center leading-tight">
        Magnetic<br />Gradient (nT)
      </div>
      <div className="flex gap-1.5">
        <canvas
          ref={canvasRef}
          style={{ width: BAR_WIDTH, height: BAR_HEIGHT }}
          className="rounded-sm"
        />
        <div className="relative" style={{ height: BAR_HEIGHT }}>
          {ticks.map(({ value, y }, i) => (
            <div
              key={i}
              className="absolute flex items-center gap-1"
              style={{ top: y, transform: 'translateY(-50%)' }}
            >
              <span className="w-1.5 h-px bg-zinc-400" />
              <span className="text-[10px] font-mono text-zinc-600 whitespace-nowrap">
                {formatTick(value)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
