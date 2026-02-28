/** Keyboard shortcut legend overlay — toggled by ? key. */

import { useAppStore } from '../stores/appStore'

const SHORTCUTS = [
  { key: '1', description: '2D view' },
  { key: '2', description: 'Split view' },
  { key: '3', description: '3D view' },
  { key: '4', description: 'Comparison view' },
  { key: '[', description: 'Toggle sidebar' },
  { key: 'L', description: 'Toggle live stream' },
  { key: 'A', description: 'Toggle anomalies' },
  { key: 'X', description: 'Cross-section mode' },
  { key: 'Esc', description: 'Cancel drawing' },
  { key: '?', description: 'Toggle this legend' },
]

export function ShortcutLegend() {
  const open = useAppStore((s) => s.shortcutLegendOpen)

  if (!open) return null

  return (
    <div className="absolute bottom-3 right-3 z-20 rounded bg-zinc-800/95 backdrop-blur-sm border border-zinc-700/50 p-3 shadow-lg">
      <h3 className="text-xs font-medium text-zinc-400 uppercase tracking-wider mb-2">
        Shortcuts
      </h3>
      <div className="space-y-1">
        {SHORTCUTS.map(({ key, description }) => (
          <div key={key} className="flex items-center gap-3 text-xs">
            <kbd className="inline-flex items-center justify-center w-6 h-5 rounded bg-zinc-700 text-zinc-300 font-mono text-[10px]">
              {key}
            </kbd>
            <span className="text-zinc-400">{description}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
