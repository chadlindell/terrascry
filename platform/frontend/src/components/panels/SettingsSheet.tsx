/** Right-side settings sheet combining ColorScaleControl and SimulationParams. */

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from '@/components/ui/sheet'
import { Separator } from '@/components/ui/separator'
import { useAppStore } from '../../stores/appStore'
import { ColorScaleControl } from '../ColorScaleControl'
import { SimulationParams } from '../SimulationParams'

export function SettingsSheet() {
  const open = useAppStore((s) => s.settingsSheetOpen)
  const setOpen = useAppStore((s) => s.setSettingsSheetOpen)

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetContent side="right" className="w-[380px] sm:max-w-[380px] overflow-y-auto">
        <SheetHeader>
          <SheetTitle>Settings</SheetTitle>
          <SheetDescription>
            Color scale and simulation parameters.
          </SheetDescription>
        </SheetHeader>
        <div className="mt-6 space-y-6">
          <section>
            <h3 className="text-sm font-medium text-zinc-900 mb-3">Color Scale</h3>
            <ColorScaleControl />
          </section>

          <Separator />

          <section>
            <h3 className="text-sm font-medium text-zinc-900 mb-3">Simulation Parameters</h3>
            <SimulationParams />
          </section>
        </div>
      </SheetContent>
    </Sheet>
  )
}
