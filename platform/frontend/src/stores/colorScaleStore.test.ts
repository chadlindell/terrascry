import { describe, it, expect, beforeEach } from 'vitest'
import { useColorScaleStore } from './colorScaleStore'

describe('colorScaleStore', () => {
  beforeEach(() => {
    useColorScaleStore.setState({
      colormap: 'viridis',
      rangeMin: 0,
      rangeMax: 1,
    })
  })

  it('has correct initial state', () => {
    const state = useColorScaleStore.getState()
    expect(state.colormap).toBe('viridis')
    expect(state.rangeMin).toBe(0)
    expect(state.rangeMax).toBe(1)
  })

  it('setColormap updates the colormap', () => {
    useColorScaleStore.getState().setColormap('plasma')
    expect(useColorScaleStore.getState().colormap).toBe('plasma')
  })

  it('setColormap supports all colormaps', () => {
    for (const name of ['viridis', 'plasma', 'inferno'] as const) {
      useColorScaleStore.getState().setColormap(name)
      expect(useColorScaleStore.getState().colormap).toBe(name)
    }
  })

  it('setRange updates min and max', () => {
    useColorScaleStore.getState().setRange(-10, 50)
    const state = useColorScaleStore.getState()
    expect(state.rangeMin).toBe(-10)
    expect(state.rangeMax).toBe(50)
  })

  it('setRange handles negative values', () => {
    useColorScaleStore.getState().setRange(-100, -5)
    const state = useColorScaleStore.getState()
    expect(state.rangeMin).toBe(-100)
    expect(state.rangeMax).toBe(-5)
  })

  it('setRange handles zero span', () => {
    useColorScaleStore.getState().setRange(42, 42)
    const state = useColorScaleStore.getState()
    expect(state.rangeMin).toBe(42)
    expect(state.rangeMax).toBe(42)
  })
})
