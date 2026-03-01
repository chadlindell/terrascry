/** Scenario selector wrapped in a Popover, anchored to the toolbar. */

import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Button } from '@/components/ui/button'
import { useAppStore } from '../../stores/appStore'
import { ScenarioSelector } from '../ScenarioSelector'
import { ChevronDown, Map } from 'lucide-react'

export function ScenarioPopover() {
  const selectedScenario = useAppStore((s) => s.selectedScenario)

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline" size="sm" className="gap-1.5 max-w-[200px]">
          <Map className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          <span className="truncate">
            {selectedScenario ?? 'Select scenario'}
          </span>
          <ChevronDown className="h-3 w-3 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-[400px] p-0 max-h-[70vh] overflow-y-auto">
        <ScenarioSelector />
      </PopoverContent>
    </Popover>
  )
}
