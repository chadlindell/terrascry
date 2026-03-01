/** Help dialog showing keyboard shortcuts. */

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogDescription,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { HelpCircle } from 'lucide-react'
import { useAppStore } from '../../stores/appStore'

const SHORTCUTS = [
  { key: '1', description: '2D view' },
  { key: '2', description: 'Split view' },
  { key: '3', description: '3D view' },
  { key: '4', description: 'Comparison view' },
  { key: 'C', description: 'Toggle contours' },
  { key: 'A', description: 'Toggle anomalies' },
  { key: 'X', description: 'Cross-section mode' },
  { key: 'L', description: 'Toggle live stream' },
  { key: '\u2318K', description: 'Command palette' },
  { key: '?', description: 'This dialog' },
  { key: 'Esc', description: 'Cancel / close' },
]

export function HelpDialog() {
  const shortcutLegendOpen = useAppStore((s) => s.shortcutLegendOpen)

  return (
    <Dialog
      open={shortcutLegendOpen}
      onOpenChange={(open) => useAppStore.setState({ shortcutLegendOpen: open })}
    >
      <DialogTrigger asChild>
        <Button variant="ghost" size="icon" className="h-8 w-8">
          <HelpCircle className="h-4 w-4" />
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[360px]">
        <DialogHeader>
          <DialogTitle>Keyboard Shortcuts</DialogTitle>
          <DialogDescription>
            Quick access keys for all TERRASCRY functions.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-1">
          {SHORTCUTS.map(({ key, description }) => (
            <div key={key} className="flex items-center justify-between py-1.5">
              <span className="text-sm text-zinc-600">{description}</span>
              <kbd className="inline-flex items-center justify-center min-w-[28px] h-6 px-1.5 rounded border border-zinc-200 bg-zinc-50 text-xs font-mono text-zinc-600">
                {key}
              </kbd>
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  )
}
