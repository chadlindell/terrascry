/** Global application state — selection, sidebar, and view mode. */

import { create } from 'zustand'

/** Active view layout mode. */
export type ViewMode = 'split' | '2d' | '3d' | 'comparison'

interface AppState {
  selectedScenario: string | null
  activeDatasetId: string | null
  sidebarOpen: boolean
  viewMode: ViewMode
  shortcutLegendOpen: boolean
  showAnomalies: boolean
  setSelectedScenario: (name: string | null) => void
  setActiveDatasetId: (id: string | null) => void
  toggleSidebar: () => void
  setViewMode: (mode: ViewMode) => void
}

/** Zustand store for global app state (scenario selection, dataset ID, sidebar, view mode). */
export const useAppStore = create<AppState>((set) => ({
  selectedScenario: null,
  activeDatasetId: null,
  sidebarOpen: true,
  viewMode: 'split',
  shortcutLegendOpen: false,
  showAnomalies: false,
  setSelectedScenario: (name) => set({ selectedScenario: name }),
  setActiveDatasetId: (id) => set({ activeDatasetId: id }),
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
  setViewMode: (mode) => set({ viewMode: mode }),
}))
