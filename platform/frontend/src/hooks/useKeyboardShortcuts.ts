/** Global keyboard shortcuts for view mode switching and UI toggles. */

import { useEffect } from 'react'
import { useAppStore } from '../stores/appStore'
import { useStreamStore } from '../stores/streamStore'
import { useCrossSectionStore } from '../stores/crossSectionStore'

/** Registers global keydown listeners. Call once at app root. */
export function useKeyboardShortcuts() {
  const setViewMode = useAppStore((s) => s.setViewMode)

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Cmd+K / Ctrl+K — command palette (works even in inputs)
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        useAppStore.setState((s) => ({ commandPaletteOpen: !s.commandPaletteOpen }))
        return
      }

      // Skip if focus is in an interactive element
      const tag = (e.target as HTMLElement)?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return

      switch (e.key) {
        case '1':
          setViewMode('2d')
          break
        case '2':
          setViewMode('split')
          break
        case '3':
          setViewMode('3d')
          break
        case '4':
          setViewMode('comparison')
          break
        case '?':
          useAppStore.setState((s) => ({ shortcutLegendOpen: !s.shortcutLegendOpen }))
          break
        case 'l':
        case 'L':
          useStreamStore.getState().toggleStream()
          break
        case 'a':
        case 'A':
          useAppStore.getState().toggleAnomalies()
          break
        case 'c':
        case 'C':
          useAppStore.getState().toggleContours()
          break
        case 'x':
        case 'X': {
          const cs = useCrossSectionStore.getState()
          if (cs.isDrawing) {
            cs.setIsDrawing(false)
          } else {
            cs.reset()
            cs.setIsDrawing(true)
          }
          break
        }
        case 'Escape':
          useCrossSectionStore.getState().setIsDrawing(false)
          useAppStore.setState({ commandPaletteOpen: false })
          break
      }
    }

    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [setViewMode])
}
