import { describe, it, expect, beforeEach } from 'vitest'
import { renderHook } from '@testing-library/react'
import { useKeyboardShortcuts } from './useKeyboardShortcuts'
import { useAppStore } from '../stores/appStore'
import { useStreamStore } from '../stores/streamStore'
import { useCrossSectionStore } from '../stores/crossSectionStore'

function fireKey(key: string, target?: EventTarget, opts?: Partial<KeyboardEventInit>) {
  const event = new KeyboardEvent('keydown', { key, bubbles: true, ...opts })
  if (target) {
    Object.defineProperty(event, 'target', { value: target })
  }
  window.dispatchEvent(event)
}

describe('useKeyboardShortcuts', () => {
  beforeEach(() => {
    useAppStore.setState({
      selectedScenario: null,
      activeDatasetId: null,
      viewMode: 'split',
      shortcutLegendOpen: false,
      showAnomalies: false,
      showContours: true,
      commandPaletteOpen: false,
      settingsSheetOpen: false,
      dataSheetOpen: false,
    })
    useStreamStore.setState({ streamEnabled: true })
    useCrossSectionStore.getState().reset()
  })

  it('1 key sets view mode to 2d', () => {
    renderHook(() => useKeyboardShortcuts())
    fireKey('1')
    expect(useAppStore.getState().viewMode).toBe('2d')
  })

  it('2 key sets view mode to split', () => {
    useAppStore.setState({ viewMode: '2d' })
    renderHook(() => useKeyboardShortcuts())
    fireKey('2')
    expect(useAppStore.getState().viewMode).toBe('split')
  })

  it('3 key sets view mode to 3d', () => {
    renderHook(() => useKeyboardShortcuts())
    fireKey('3')
    expect(useAppStore.getState().viewMode).toBe('3d')
  })

  it('4 key sets view mode to comparison', () => {
    renderHook(() => useKeyboardShortcuts())
    fireKey('4')
    expect(useAppStore.getState().viewMode).toBe('comparison')
  })

  it('Cmd+K toggles command palette', () => {
    renderHook(() => useKeyboardShortcuts())
    expect(useAppStore.getState().commandPaletteOpen).toBe(false)
    fireKey('k', undefined, { metaKey: true })
    expect(useAppStore.getState().commandPaletteOpen).toBe(true)
    fireKey('k', undefined, { metaKey: true })
    expect(useAppStore.getState().commandPaletteOpen).toBe(false)
  })

  it('Ctrl+K toggles command palette', () => {
    renderHook(() => useKeyboardShortcuts())
    expect(useAppStore.getState().commandPaletteOpen).toBe(false)
    fireKey('k', undefined, { ctrlKey: true })
    expect(useAppStore.getState().commandPaletteOpen).toBe(true)
  })

  it('Cmd+K works even when INPUT is focused', () => {
    renderHook(() => useKeyboardShortcuts())
    const input = { tagName: 'INPUT' } as HTMLElement
    fireKey('k', input, { metaKey: true })
    expect(useAppStore.getState().commandPaletteOpen).toBe(true)
  })

  it('? key toggles shortcut legend', () => {
    renderHook(() => useKeyboardShortcuts())
    expect(useAppStore.getState().shortcutLegendOpen).toBe(false)
    fireKey('?')
    expect(useAppStore.getState().shortcutLegendOpen).toBe(true)
    fireKey('?')
    expect(useAppStore.getState().shortcutLegendOpen).toBe(false)
  })

  it('C key toggles contours', () => {
    renderHook(() => useKeyboardShortcuts())
    expect(useAppStore.getState().showContours).toBe(true)
    fireKey('C')
    expect(useAppStore.getState().showContours).toBe(false)
  })

  it('c key (lowercase) toggles contours', () => {
    renderHook(() => useKeyboardShortcuts())
    expect(useAppStore.getState().showContours).toBe(true)
    fireKey('c')
    expect(useAppStore.getState().showContours).toBe(false)
    fireKey('c')
    expect(useAppStore.getState().showContours).toBe(true)
  })

  it('A key toggles anomalies', () => {
    renderHook(() => useKeyboardShortcuts())
    expect(useAppStore.getState().showAnomalies).toBe(false)
    fireKey('A')
    expect(useAppStore.getState().showAnomalies).toBe(true)
    fireKey('a')
    expect(useAppStore.getState().showAnomalies).toBe(false)
  })

  it('Escape closes command palette', () => {
    useAppStore.setState({ commandPaletteOpen: true })
    renderHook(() => useKeyboardShortcuts())
    fireKey('Escape')
    expect(useAppStore.getState().commandPaletteOpen).toBe(false)
  })

  it('ignores keypress when INPUT is focused', () => {
    renderHook(() => useKeyboardShortcuts())
    const input = { tagName: 'INPUT' } as HTMLElement
    fireKey('1', input)
    // viewMode should remain unchanged
    expect(useAppStore.getState().viewMode).toBe('split')
  })

  it('cleans up event listener on unmount', () => {
    const { unmount } = renderHook(() => useKeyboardShortcuts())
    unmount()
    fireKey('3')
    // After unmount, keydown should have no effect — viewMode stays at split
    expect(useAppStore.getState().viewMode).toBe('split')
  })
})
