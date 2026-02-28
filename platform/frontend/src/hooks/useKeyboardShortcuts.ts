/** Global keyboard shortcuts for view mode switching and UI toggles. */

import { useEffect } from 'react'
import { useAppStore } from '../stores/appStore'

/** Registers global keydown listeners. Call once at app root. */
export function useKeyboardShortcuts() {
  const setViewMode = useAppStore((s) => s.setViewMode)
  const toggleSidebar = useAppStore((s) => s.toggleSidebar)

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
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
        case '[':
          toggleSidebar()
          break
        case '?':
          useAppStore.setState((s) => ({ shortcutLegendOpen: !s.shortcutLegendOpen }))
          break
      }
    }

    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [setViewMode, toggleSidebar])
}
