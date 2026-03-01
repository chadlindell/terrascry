import { describe, it, expect, beforeEach } from 'vitest'
import { useAppStore } from './appStore'

describe('appStore', () => {
  beforeEach(() => {
    useAppStore.setState({
      selectedScenario: null,
      activeDatasetId: null,
      viewMode: 'split',
      showContours: true,
      showAnomalies: false,
      shortcutLegendOpen: false,
      commandPaletteOpen: false,
      settingsSheetOpen: false,
      dataSheetOpen: false,
    })
  })

  it('has correct initial state', () => {
    const state = useAppStore.getState()
    expect(state.selectedScenario).toBeNull()
    expect(state.activeDatasetId).toBeNull()
    expect(state.viewMode).toBe('split')
    expect(state.commandPaletteOpen).toBe(false)
    expect(state.settingsSheetOpen).toBe(false)
    expect(state.dataSheetOpen).toBe(false)
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

  it('setViewMode updates viewMode', () => {
    useAppStore.getState().setViewMode('2d')
    expect(useAppStore.getState().viewMode).toBe('2d')
    useAppStore.getState().setViewMode('3d')
    expect(useAppStore.getState().viewMode).toBe('3d')
    useAppStore.getState().setViewMode('split')
    expect(useAppStore.getState().viewMode).toBe('split')
  })

  it('showContours defaults to true', () => {
    expect(useAppStore.getState().showContours).toBe(true)
  })

  it('toggleContours flips showContours', () => {
    expect(useAppStore.getState().showContours).toBe(true)
    useAppStore.getState().toggleContours()
    expect(useAppStore.getState().showContours).toBe(false)
    useAppStore.getState().toggleContours()
    expect(useAppStore.getState().showContours).toBe(true)
  })

  it('toggleAnomalies flips showAnomalies', () => {
    expect(useAppStore.getState().showAnomalies).toBe(false)
    useAppStore.getState().toggleAnomalies()
    expect(useAppStore.getState().showAnomalies).toBe(true)
    useAppStore.getState().toggleAnomalies()
    expect(useAppStore.getState().showAnomalies).toBe(false)
  })

  it('setCommandPaletteOpen controls command palette', () => {
    useAppStore.getState().setCommandPaletteOpen(true)
    expect(useAppStore.getState().commandPaletteOpen).toBe(true)
    useAppStore.getState().setCommandPaletteOpen(false)
    expect(useAppStore.getState().commandPaletteOpen).toBe(false)
  })

  it('toggleCommandPalette flips commandPaletteOpen', () => {
    expect(useAppStore.getState().commandPaletteOpen).toBe(false)
    useAppStore.getState().toggleCommandPalette()
    expect(useAppStore.getState().commandPaletteOpen).toBe(true)
    useAppStore.getState().toggleCommandPalette()
    expect(useAppStore.getState().commandPaletteOpen).toBe(false)
  })

  it('setShortcutLegendOpen controls shortcut legend', () => {
    useAppStore.getState().setShortcutLegendOpen(true)
    expect(useAppStore.getState().shortcutLegendOpen).toBe(true)
    useAppStore.getState().setShortcutLegendOpen(false)
    expect(useAppStore.getState().shortcutLegendOpen).toBe(false)
  })

  it('toggleShortcutLegend flips shortcutLegendOpen', () => {
    expect(useAppStore.getState().shortcutLegendOpen).toBe(false)
    useAppStore.getState().toggleShortcutLegend()
    expect(useAppStore.getState().shortcutLegendOpen).toBe(true)
    useAppStore.getState().toggleShortcutLegend()
    expect(useAppStore.getState().shortcutLegendOpen).toBe(false)
  })

  it('setSettingsSheetOpen controls settings sheet', () => {
    useAppStore.getState().setSettingsSheetOpen(true)
    expect(useAppStore.getState().settingsSheetOpen).toBe(true)
    useAppStore.getState().setSettingsSheetOpen(false)
    expect(useAppStore.getState().settingsSheetOpen).toBe(false)
  })

  it('setDataSheetOpen controls data sheet', () => {
    useAppStore.getState().setDataSheetOpen(true)
    expect(useAppStore.getState().dataSheetOpen).toBe(true)
    useAppStore.getState().setDataSheetOpen(false)
    expect(useAppStore.getState().dataSheetOpen).toBe(false)
  })
})
