import { describe, it, expect, beforeEach } from 'vitest'
import { useAppStore } from './appStore'

describe('appStore', () => {
  beforeEach(() => {
    useAppStore.setState({
      selectedScenario: null,
      activeDatasetId: null,
      sidebarOpen: true,
      viewMode: 'split',
    })
  })

  it('has correct initial state', () => {
    const state = useAppStore.getState()
    expect(state.selectedScenario).toBeNull()
    expect(state.activeDatasetId).toBeNull()
    expect(state.sidebarOpen).toBe(true)
    expect(state.viewMode).toBe('split')
  })

  it('setSelectedScenario updates selected scenario', () => {
    useAppStore.getState().setSelectedScenario('test-scenario')
    expect(useAppStore.getState().selectedScenario).toBe('test-scenario')
  })

  it('setSelectedScenario can clear to null', () => {
    useAppStore.getState().setSelectedScenario('test-scenario')
    useAppStore.getState().setSelectedScenario(null)
    expect(useAppStore.getState().selectedScenario).toBeNull()
  })

  it('setActiveDatasetId updates active dataset', () => {
    useAppStore.getState().setActiveDatasetId('abc-123')
    expect(useAppStore.getState().activeDatasetId).toBe('abc-123')
  })

  it('setActiveDatasetId can clear to null', () => {
    useAppStore.getState().setActiveDatasetId('abc-123')
    useAppStore.getState().setActiveDatasetId(null)
    expect(useAppStore.getState().activeDatasetId).toBeNull()
  })

  it('toggleSidebar flips sidebarOpen', () => {
    expect(useAppStore.getState().sidebarOpen).toBe(true)
    useAppStore.getState().toggleSidebar()
    expect(useAppStore.getState().sidebarOpen).toBe(false)
    useAppStore.getState().toggleSidebar()
    expect(useAppStore.getState().sidebarOpen).toBe(true)
  })

  it('setViewMode updates viewMode', () => {
    useAppStore.getState().setViewMode('2d')
    expect(useAppStore.getState().viewMode).toBe('2d')
    useAppStore.getState().setViewMode('3d')
    expect(useAppStore.getState().viewMode).toBe('3d')
    useAppStore.getState().setViewMode('split')
    expect(useAppStore.getState().viewMode).toBe('split')
  })
})
