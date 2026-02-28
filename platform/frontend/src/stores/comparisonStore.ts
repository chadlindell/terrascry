/** Zustand store for dataset comparison state. */

import { create } from 'zustand'

export type ComparisonMode = 'side-by-side' | 'difference'

interface ComparisonState {
  comparisonDatasetId: string | null
  mode: ComparisonMode
  opacity: number
  setComparisonDatasetId: (id: string | null) => void
  setMode: (mode: ComparisonMode) => void
  setOpacity: (opacity: number) => void
}

export const useComparisonStore = create<ComparisonState>((set) => ({
  comparisonDatasetId: null,
  mode: 'side-by-side',
  opacity: 1.0,
  setComparisonDatasetId: (id) => set({ comparisonDatasetId: id }),
  setMode: (mode) => set({ mode }),
  setOpacity: (opacity) => set({ opacity }),
}))
