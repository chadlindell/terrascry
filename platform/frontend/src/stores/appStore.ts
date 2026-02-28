import { create } from 'zustand'

export type ViewMode = 'split' | '2d' | '3d'

interface AppState {
  selectedScenario: string | null
  activeDatasetId: string | null
  sidebarOpen: boolean
  viewMode: ViewMode
  setSelectedScenario: (name: string | null) => void
  setActiveDatasetId: (id: string | null) => void
  toggleSidebar: () => void
  setViewMode: (mode: ViewMode) => void
}

export const useAppStore = create<AppState>((set) => ({
  selectedScenario: null,
  activeDatasetId: null,
  sidebarOpen: true,
  viewMode: 'split',
  setSelectedScenario: (name) => set({ selectedScenario: name }),
  setActiveDatasetId: (id) => set({ activeDatasetId: id }),
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
  setViewMode: (mode) => set({ viewMode: mode }),
}))
