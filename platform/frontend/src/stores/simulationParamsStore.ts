/** Simulation parameter state — controls sent with survey requests. */

import { create } from 'zustand'

interface SimulationParamsState {
  lineSpacing: number
  sampleSpacing: number
  resolution: number
  setLineSpacing: (v: number) => void
  setSampleSpacing: (v: number) => void
  setResolution: (v: number) => void
}

export const useSimulationParamsStore = create<SimulationParamsState>((set) => ({
  lineSpacing: 1.0,
  sampleSpacing: 0.5,
  resolution: 0.5,
  setLineSpacing: (v) => set({ lineSpacing: v }),
  setSampleSpacing: (v) => set({ sampleSpacing: v }),
  setResolution: (v) => set({ resolution: v }),
}))
