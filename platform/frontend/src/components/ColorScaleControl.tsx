/** Color scale control panel — colormap selector, range inputs, and gradient preview. */

import { useRef, useEffect } from 'react'
import { useColorScaleStore } from '../stores/colorScaleStore'
import { COLORMAPS, type ColormapName } from '../colormap'

const COLORMAP_OPTIONS = Object.keys(COLORMAPS) as ColormapName[]

function ColorGradientBar({ colormap }: { colormap: ColormapName }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height
    const lut = COLORMAPS[colormap]

    for (let x = 0; x < width; x++) {
      const i = Math.round((x / (width - 1)) * 255)
      const [r, g, b] = lut[i]
      ctx.fillStyle = `rgb(${r},${g},${b})`
      ctx.fillRect(x, 0, 1, height)
    }
  }, [colormap])

  return (
    <canvas
      ref={canvasRef}
      width={200}
      height={12}
      className="w-full h-3 rounded-sm"
    />
  )
}

export function ColorScaleControl() {
  const colormap = useColorScaleStore((s) => s.colormap)
  const rangeMin = useColorScaleStore((s) => s.rangeMin)
  const rangeMax = useColorScaleStore((s) => s.rangeMax)
  const setColormap = useColorScaleStore((s) => s.setColormap)
  const setRange = useColorScaleStore((s) => s.setRange)

  return (
    <div className="space-y-2">
      <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-[0.08em]">
        Color Scale
      </span>

      {/* Gradient preview */}
      <ColorGradientBar colormap={colormap} />

      {/* Colormap selector */}
      <div>
        <label className="text-xs text-zinc-400 block mb-1">Palette</label>
        <select
          value={colormap}
          onChange={(e) => setColormap(e.target.value as ColormapName)}
          className="w-full bg-white text-zinc-800 text-xs rounded-lg px-2 py-1.5 border border-zinc-300 transition-all duration-150 focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 focus:outline-none"
        >
          {COLORMAP_OPTIONS.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </div>

      {/* Range inputs */}
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="text-xs text-zinc-400 block mb-1">Min (nT)</label>
          <input
            type="number"
            value={rangeMin.toFixed(1)}
            onChange={(e) => setRange(parseFloat(e.target.value) || 0, rangeMax)}
            className="w-full bg-white text-zinc-800 text-xs rounded-lg px-2 py-1.5 border border-zinc-300 transition-all duration-150 focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 focus:outline-none"
          />
        </div>
        <div>
          <label className="text-xs text-zinc-400 block mb-1">Max (nT)</label>
          <input
            type="number"
            value={rangeMax.toFixed(1)}
            onChange={(e) => setRange(rangeMin, parseFloat(e.target.value) || 0)}
            className="w-full bg-white text-zinc-800 text-xs rounded-lg px-2 py-1.5 border border-zinc-300 transition-all duration-150 focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 focus:outline-none"
          />
        </div>
      </div>
    </div>
  )
}
