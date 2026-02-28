/** Zustand store for real-time streaming state. */

import { create } from 'zustand'
import type { StreamPoint, AnomalyEvent } from '../types/streaming'

export type StreamStatus = 'disconnected' | 'connecting' | 'connected' | 'error'

const MAX_POINTS = 500

interface StreamState {
  status: StreamStatus
  points: StreamPoint[]
  anomalies: AnomalyEvent[]
  messageRate: number
  streamEnabled: boolean
  setStatus: (status: StreamStatus) => void
  addPoint: (point: StreamPoint) => void
  addAnomaly: (anomaly: AnomalyEvent) => void
  clearPoints: () => void
  updateRate: (rate: number) => void
  toggleStream: () => void
}

export const useStreamStore = create<StreamState>((set) => ({
  status: 'disconnected',
  points: [],
  anomalies: [],
  messageRate: 0,
  streamEnabled: true,
  setStatus: (status) => set({ status }),
  addPoint: (point) =>
    set((s) => ({
      points:
        s.points.length >= MAX_POINTS
          ? [...s.points.slice(s.points.length - MAX_POINTS + 1), point]
          : [...s.points, point],
    })),
  addAnomaly: (anomaly) => set((s) => ({ anomalies: [...s.anomalies, anomaly] })),
  clearPoints: () => set({ points: [], anomalies: [] }),
  updateRate: (rate) => set({ messageRate: rate }),
  toggleStream: () => set((s) => ({ streamEnabled: !s.streamEnabled })),
}))
