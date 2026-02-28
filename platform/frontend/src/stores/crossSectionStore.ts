/** Zustand store for cross-section profile drawing and data. */

import { create } from 'zustand'

export interface ProfilePoint {
  distance: number
  x: number
  y: number
  gradient_nt: number
}

interface CrossSectionState {
  startPoint: [number, number] | null
  endPoint: [number, number] | null
  isDrawing: boolean
  profileData: ProfilePoint[]
  cursorPosition: number | null
  setStartPoint: (pt: [number, number] | null) => void
  setEndPoint: (pt: [number, number] | null) => void
  setIsDrawing: (drawing: boolean) => void
  setProfileData: (data: ProfilePoint[]) => void
  setCursorPosition: (pos: number | null) => void
  reset: () => void
}

export const useCrossSectionStore = create<CrossSectionState>((set) => ({
  startPoint: null,
  endPoint: null,
  isDrawing: false,
  profileData: [],
  cursorPosition: null,
  setStartPoint: (pt) => set({ startPoint: pt }),
  setEndPoint: (pt) => set({ endPoint: pt }),
  setIsDrawing: (drawing) => set({ isDrawing: drawing }),
  setProfileData: (data) => set({ profileData: data }),
  setCursorPosition: (pos) => set({ cursorPosition: pos }),
  reset: () =>
    set({
      startPoint: null,
      endPoint: null,
      isDrawing: false,
      profileData: [],
      cursorPosition: null,
    }),
}))
