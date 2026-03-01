import { describe, it, expect, beforeEach } from 'vitest'
import { useStreamStore } from './streamStore'
import type { StreamPoint, AnomalyEvent } from '../types/streaming'

const mockPoint: StreamPoint = {
  x: 1.0,
  y: 2.0,
  gradient_nt: 42.5,
  timestamp: '2026-03-01T00:00:00Z',
}

const mockAnomaly: AnomalyEvent = {
  x: 3.0,
  y: 4.0,
  anomaly_strength_nt: 100.0,
  anomaly_type: 'dipole',
  confidence: 0.95,
  timestamp: '2026-03-01T00:00:00Z',
}

describe('streamStore', () => {
  beforeEach(() => {
    useStreamStore.setState({
      status: 'disconnected',
      points: [],
      anomalies: [],
      messageRate: 0,
      streamEnabled: true,
    })
  })

  it('has correct defaults', () => {
    const state = useStreamStore.getState()
    expect(state.status).toBe('disconnected')
    expect(state.points).toEqual([])
    expect(state.anomalies).toEqual([])
    expect(state.messageRate).toBe(0)
    expect(state.streamEnabled).toBe(true)
  })

  it('setStatus updates connection status', () => {
    useStreamStore.getState().setStatus('connected')
    expect(useStreamStore.getState().status).toBe('connected')
  })

  it('addPoint appends a stream point', () => {
    useStreamStore.getState().addPoint(mockPoint)
    expect(useStreamStore.getState().points).toHaveLength(1)
    expect(useStreamStore.getState().points[0]).toEqual(mockPoint)
  })

  it('addPoint caps at 500 points', () => {
    const store = useStreamStore.getState()
    for (let i = 0; i < 502; i++) {
      store.addPoint({ ...mockPoint, x: i })
    }
    const points = useStreamStore.getState().points
    expect(points.length).toBe(500)
    // Oldest points should be trimmed — first kept point is x=2
    expect(points[0].x).toBe(2)
    expect(points[points.length - 1].x).toBe(501)
  })

  it('addAnomaly appends an anomaly event', () => {
    useStreamStore.getState().addAnomaly(mockAnomaly)
    expect(useStreamStore.getState().anomalies).toHaveLength(1)
    expect(useStreamStore.getState().anomalies[0]).toEqual(mockAnomaly)
  })

  it('clearPoints resets points and anomalies', () => {
    useStreamStore.getState().addPoint(mockPoint)
    useStreamStore.getState().addAnomaly(mockAnomaly)
    useStreamStore.getState().clearPoints()
    expect(useStreamStore.getState().points).toEqual([])
    expect(useStreamStore.getState().anomalies).toEqual([])
  })

  it('updateRate updates message rate', () => {
    useStreamStore.getState().updateRate(10.5)
    expect(useStreamStore.getState().messageRate).toBe(10.5)
  })

  it('toggleStream flips streamEnabled', () => {
    expect(useStreamStore.getState().streamEnabled).toBe(true)
    useStreamStore.getState().toggleStream()
    expect(useStreamStore.getState().streamEnabled).toBe(false)
    useStreamStore.getState().toggleStream()
    expect(useStreamStore.getState().streamEnabled).toBe(true)
  })
})
