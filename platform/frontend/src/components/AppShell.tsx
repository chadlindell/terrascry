/** Root layout — top toolbar + full-height content area. */

import { Toolbar } from './toolbar/Toolbar'
import { CommandPalette } from './CommandPalette'
import { SettingsSheet } from './panels/SettingsSheet'
import { DataSheet } from './panels/DataSheet'
import { SplitWorkspace } from './SplitWorkspace'
import { TooltipProvider } from '@/components/ui/tooltip'
import { useKeyboardShortcuts } from '../hooks/useKeyboardShortcuts'

export function AppShell() {
  useKeyboardShortcuts()

  return (
    <TooltipProvider>
      <div className="flex flex-col h-screen overflow-hidden bg-white text-zinc-900 font-sans antialiased">
        <Toolbar />
        <main className="relative flex-1 min-h-0 bg-zinc-100 overflow-hidden">
          <SplitWorkspace />
        </main>

        {/* Overlays — command palette and slide-out sheets */}
        <CommandPalette />
        <SettingsSheet />
        <DataSheet />
      </div>
    </TooltipProvider>
  )
}
