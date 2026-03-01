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
        <label className="text-xs text-zinc-500">{label}</label>
        <span className="text-xs font-mono text-zinc-700">{value.toFixed(1)}m</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
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
      <h3 className="text-[11px] font-semibold text-zinc-400 uppercase tracking-[0.08em]">
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
