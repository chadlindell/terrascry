/** Global application state — selection, panels, and view mode. */

import { create } from 'zustand'

/** Active view layout mode. */
export type ViewMode = 'split' | '2d' | '3d' | 'comparison'

interface AppState {
  selectedScenario: string | null
  activeDatasetId: string | null
  viewMode: ViewMode
  shortcutLegendOpen: boolean
  showAnomalies: boolean
  showContours: boolean

  // Panel open/close state (replaces sidebar)
  commandPaletteOpen: boolean
  settingsSheetOpen: boolean
  dataSheetOpen: boolean

  setSelectedScenario: (name: string | null) => void
  setActiveDatasetId: (id: string | null) => void
  setViewMode: (mode: ViewMode) => void
  toggleContours: () => void
  toggleAnomalies: () => void
  setCommandPaletteOpen: (open: boolean) => void
  setSettingsSheetOpen: (open: boolean) => void
  setDataSheetOpen: (open: boolean) => void
}

/** Zustand store for global app state (scenario selection, dataset ID, panels, view mode). */
export const useAppStore = create<AppState>((set) => ({
  selectedScenario: null,
  activeDatasetId: null,
  viewMode: 'split',
  shortcutLegendOpen: false,
  showAnomalies: false,
  showContours: true,

  commandPaletteOpen: false,
  settingsSheetOpen: false,
  dataSheetOpen: false,

  setSelectedScenario: (name) => set({ selectedScenario: name }),
  setActiveDatasetId: (id) => set({ activeDatasetId: id }),
  setViewMode: (mode) => set({ viewMode: mode }),
  toggleContours: () => set((s) => ({ showContours: !s.showContours })),
  toggleAnomalies: () => set((s) => ({ showAnomalies: !s.showAnomalies })),
  setCommandPaletteOpen: (open) => set({ commandPaletteOpen: open }),
  setSettingsSheetOpen: (open) => set({ settingsSheetOpen: open }),
  setDataSheetOpen: (open) => set({ dataSheetOpen: open }),
}))
