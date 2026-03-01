/** Zustand store for animated survey playback state. */

import { create } from 'zustand'

interface PlaybackState {
  isPlaying: boolean
  currentIndex: number
  speed: number // points per second
  totalPoints: number

  play: () => void
  pause: () => void
  reset: () => void
  setIndex: (index: number) => void
  setSpeed: (speed: number) => void
  setTotalPoints: (total: number) => void
  advance: () => boolean // returns true if advanced, false if at end
}

export const usePlaybackStore = create<PlaybackState>((set, get) => ({
  isPlaying: false,
  currentIndex: 0,
  speed: 25,
  totalPoints: 0,

  play: () => set({ isPlaying: true }),
  pause: () => set({ isPlaying: false }),
  reset: () => set({ isPlaying: false, currentIndex: 0 }),
  setIndex: (index) => set({ currentIndex: Math.max(0, index) }),
  setSpeed: (speed) => set({ speed }),
  setTotalPoints: (total) => set({ totalPoints: total }),
  advance: () => {
    const { currentIndex, totalPoints } = get()
    if (currentIndex >= totalPoints - 1) {
      set({ isPlaying: false })
      return false
    }
    set({ currentIndex: currentIndex + 1 })
    return true
  },
}))
