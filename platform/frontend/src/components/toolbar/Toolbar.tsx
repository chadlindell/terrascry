/** Top toolbar replacing the sidebar — 48px height, all controls inline or in popovers/dropdowns. */

import { useAppStore } from '../../stores/appStore'
import { Separator } from '@/components/ui/separator'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Database, Settings, Command } from 'lucide-react'
import { ScenarioPopover } from './ScenarioPopover'
import { RunSurveyAction } from './RunSurveyAction'
import { ViewModeToggle } from './ViewModeToggle'
import { OverlaysMenu } from './OverlaysMenu'
import { StreamIndicator } from './StreamIndicator'
import { ImportDialog } from './ImportDialog'
import { ExportMenu } from './ExportMenu'
import { HelpDialog } from './HelpDialog'

export function Toolbar() {
  const setCommandPaletteOpen = useAppStore((s) => s.setCommandPaletteOpen)
  const setSettingsSheetOpen = useAppStore((s) => s.setSettingsSheetOpen)
  const setDataSheetOpen = useAppStore((s) => s.setDataSheetOpen)

  return (
    <header className="flex items-center h-12 px-3 gap-2 border-b border-border bg-white/80 backdrop-blur-sm shrink-0">
      {/* Left group — Brand + Data */}
      <span className="text-sm font-semibold tracking-tight text-zinc-900 mr-1">
        TERRASCRY
      </span>

      <ScenarioPopover />
      <RunSurveyAction />

      <Separator orientation="vertical" className="h-6 mx-1" />

      {/* Center — Views */}
      <ViewModeToggle />

      <Separator orientation="vertical" className="h-6 mx-1" />

      {/* Center-right — Overlays + Live */}
      <OverlaysMenu />
      <StreamIndicator />

      {/* Spacer */}
      <div className="flex-1" />

      {/* Right — Settings + I/O */}
      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8"
        onClick={() => setSettingsSheetOpen(true)}
      >
        <Settings className="h-4 w-4" />
      </Button>

      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8"
        onClick={() => setDataSheetOpen(true)}
      >
        <Database className="h-4 w-4" />
      </Button>

      <ImportDialog />
      <ExportMenu />

      <Button
        variant="outline"
        size="sm"
        className="gap-1 h-7 px-2"
        onClick={() => setCommandPaletteOpen(true)}
      >
        <Command className="h-3 w-3" />
        <Badge variant="secondary" className="px-1 py-0 text-[10px] font-normal">
          K
        </Badge>
      </Button>

      <HelpDialog />
    </header>
  )
}
