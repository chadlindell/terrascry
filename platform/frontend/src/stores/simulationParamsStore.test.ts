import { describe, it, expect, beforeEach } from 'vitest'
import { useSimulationParamsStore } from './simulationParamsStore'

describe('simulationParamsStore', () => {
  beforeEach(() => {
    useSimulationParamsStore.setState({
      lineSpacing: 1.0,
      sampleSpacing: 0.5,
      resolution: 0.5,
    })
  })

  it('has correct defaults', () => {
    const state = useSimulationParamsStore.getState()
    expect(state.lineSpacing).toBe(1.0)
    expect(state.sampleSpacing).toBe(0.5)
    expect(state.resolution).toBe(0.5)
  })

  it('setLineSpacing updates line spacing', () => {
    useSimulationParamsStore.getState().setLineSpacing(2.0)
    expect(useSimulationParamsStore.getState().lineSpacing).toBe(2.0)
  })

  it('setSampleSpacing updates sample spacing', () => {
    useSimulationParamsStore.getState().setSampleSpacing(0.25)
    expect(useSimulationParamsStore.getState().sampleSpacing).toBe(0.25)
  })

  it('setResolution updates resolution', () => {
    useSimulationParamsStore.getState().setResolution(1.0)
    expect(useSimulationParamsStore.getState().resolution).toBe(1.0)
  })
})
