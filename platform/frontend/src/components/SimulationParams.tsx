/** Simulation parameter sliders for line spacing, sample spacing, and resolution. */

import { useSimulationParamsStore } from '../stores/simulationParamsStore'

interface ParamSliderProps {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
}

function ParamSlider({ label, value, min, max, step, onChange }: ParamSliderProps) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <label className="text-xs text-zinc-400">{label}</label>
        <span className="text-xs font-mono text-zinc-300">{value.toFixed(1)}m</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1 bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
      />
    </div>
  )
}

export function SimulationParams() {
  const lineSpacing = useSimulationParamsStore((s) => s.lineSpacing)
  const sampleSpacing = useSimulationParamsStore((s) => s.sampleSpacing)
  const resolution = useSimulationParamsStore((s) => s.resolution)
  const setLineSpacing = useSimulationParamsStore((s) => s.setLineSpacing)
  const setSampleSpacing = useSimulationParamsStore((s) => s.setSampleSpacing)
  const setResolution = useSimulationParamsStore((s) => s.setResolution)

  return (
    <div className="space-y-2">
      <h3 className="text-xs font-medium text-zinc-400 uppercase tracking-wider">
        Parameters
      </h3>
      <ParamSlider
        label="Line Spacing"
        value={lineSpacing}
        min={0.1}
        max={10}
        step={0.1}
        onChange={setLineSpacing}
      />
      <ParamSlider
        label="Sample Spacing"
        value={sampleSpacing}
        min={0.1}
        max={5}
        step={0.1}
        onChange={setSampleSpacing}
      />
      <ParamSlider
        label="Resolution"
        value={resolution}
        min={0.1}
        max={5}
        step={0.1}
        onChange={setResolution}
      />
    </div>
  )
}
