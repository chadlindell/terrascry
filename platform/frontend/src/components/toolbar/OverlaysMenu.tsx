/** Overlays dropdown menu with checkbox items for contours, anomalies, cross-section. */

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuCheckboxItem,
  DropdownMenuTrigger,
  DropdownMenuShortcut,
} from '@/components/ui/dropdown-menu'
import { Button } from '@/components/ui/button'
import { Layers, ChevronDown } from 'lucide-react'
import { useAppStore } from '../../stores/appStore'
import { useCrossSectionStore } from '../../stores/crossSectionStore'

export function OverlaysMenu() {
  const showContours = useAppStore((s) => s.showContours)
  const showAnomalies = useAppStore((s) => s.showAnomalies)
  const csIsDrawing = useCrossSectionStore((s) => s.isDrawing)

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className="gap-1.5">
          <Layers className="h-3.5 w-3.5" />
          Overlays
          <ChevronDown className="h-3 w-3 opacity-50" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        <DropdownMenuCheckboxItem
          checked={showContours}
          onCheckedChange={() => useAppStore.getState().toggleContours()}
        >
          Contours
          <DropdownMenuShortcut>C</DropdownMenuShortcut>
        </DropdownMenuCheckboxItem>
        <DropdownMenuCheckboxItem
          checked={showAnomalies}
          onCheckedChange={() =>
            useAppStore.getState().toggleAnomalies()
          }
        >
          Anomalies
          <DropdownMenuShortcut>A</DropdownMenuShortcut>
        </DropdownMenuCheckboxItem>
        <DropdownMenuCheckboxItem
          checked={csIsDrawing}
          onCheckedChange={(checked) => {
            const cs = useCrossSectionStore.getState()
            if (checked) {
              cs.reset()
              cs.setIsDrawing(true)
            } else {
              cs.setIsDrawing(false)
            }
          }}
        >
          Cross-section mode
          <DropdownMenuShortcut>X</DropdownMenuShortcut>
        </DropdownMenuCheckboxItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
