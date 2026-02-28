import { create } from 'zustand'

interface AppState {
  selectedScenario: string | null
  activeDatasetId: string | null
  sidebarOpen: boolean
  setSelectedScenario: (name: string | null) => void
  setActiveDatasetId: (id: string | null) => void
  toggleSidebar: () => void
}

export const useAppStore = create<AppState>((set) => ({
  selectedScenario: null,
  activeDatasetId: null,
  sidebarOpen: true,
  setSelectedScenario: (name) => set({ selectedScenario: name }),
  setActiveDatasetId: (id) => set({ activeDatasetId: id }),
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
}))
