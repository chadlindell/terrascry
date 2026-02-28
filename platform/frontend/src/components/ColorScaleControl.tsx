/** Color scale control panel — colormap selector and range inputs. */

import { useColorScaleStore } from '../stores/colorScaleStore'
import { COLORMAPS, type ColormapName } from '../colormap'

const COLORMAP_OPTIONS = Object.keys(COLORMAPS) as ColormapName[]

export function ColorScaleControl() {
  const colormap = useColorScaleStore((s) => s.colormap)
  const rangeMin = useColorScaleStore((s) => s.rangeMin)
  const rangeMax = useColorScaleStore((s) => s.rangeMax)
  const setColormap = useColorScaleStore((s) => s.setColormap)
  const setRange = useColorScaleStore((s) => s.setRange)

  return (
    <div className="space-y-2">
      <span className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">
        Color Scale
      </span>

      {/* Colormap selector */}
      <div>
        <label className="text-xs text-zinc-400 block mb-1">Palette</label>
        <select
          value={colormap}
          onChange={(e) => setColormap(e.target.value as ColormapName)}
          className="w-full bg-white text-zinc-800 text-xs rounded px-2 py-1.5 border border-zinc-300 focus:border-emerald-500 focus:outline-none"
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
            className="w-full bg-white text-zinc-800 text-xs rounded px-2 py-1.5 border border-zinc-300 focus:border-emerald-500 focus:outline-none"
          />
        </div>
        <div>
          <label className="text-xs text-zinc-400 block mb-1">Max (nT)</label>
          <input
            type="number"
            value={rangeMax.toFixed(1)}
            onChange={(e) => setRange(rangeMin, parseFloat(e.target.value) || 0)}
            className="w-full bg-white text-zinc-800 text-xs rounded px-2 py-1.5 border border-zinc-300 focus:border-emerald-500 focus:outline-none"
          />
        </div>
      </div>
    </div>
  )
}
