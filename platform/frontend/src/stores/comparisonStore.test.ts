import { describe, it, expect, beforeEach } from 'vitest'
import { useComparisonStore } from './comparisonStore'

describe('comparisonStore', () => {
  beforeEach(() => {
    useComparisonStore.setState({
      comparisonDatasetId: null,
      mode: 'side-by-side',
      opacity: 1.0,
    })
  })

  it('has correct defaults', () => {
    const state = useComparisonStore.getState()
    expect(state.comparisonDatasetId).toBeNull()
    expect(state.mode).toBe('side-by-side')
    expect(state.opacity).toBe(1.0)
  })

  it('setComparisonDatasetId updates dataset id', () => {
    useComparisonStore.getState().setComparisonDatasetId('abc-123')
    expect(useComparisonStore.getState().comparisonDatasetId).toBe('abc-123')
  })

  it('setComparisonDatasetId can clear to null', () => {
    useComparisonStore.getState().setComparisonDatasetId('abc-123')
    useComparisonStore.getState().setComparisonDatasetId(null)
    expect(useComparisonStore.getState().comparisonDatasetId).toBeNull()
  })

  it('setMode switches comparison mode', () => {
    useComparisonStore.getState().setMode('difference')
    expect(useComparisonStore.getState().mode).toBe('difference')
    useComparisonStore.getState().setMode('side-by-side')
    expect(useComparisonStore.getState().mode).toBe('side-by-side')
  })

  it('setOpacity updates opacity', () => {
    useComparisonStore.getState().setOpacity(0.5)
    expect(useComparisonStore.getState().opacity).toBe(0.5)
  })
})
