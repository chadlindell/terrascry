/** Right-side data sheet with dataset history. */

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from '@/components/ui/sheet'
import { useAppStore } from '../../stores/appStore'
import { DatasetHistory } from '../DatasetHistory'

export function DataSheet() {
  const open = useAppStore((s) => s.dataSheetOpen)
  const setOpen = useAppStore((s) => s.setDataSheetOpen)

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetContent side="right" className="w-[380px] sm:max-w-[380px] overflow-y-auto">
        <SheetHeader>
          <SheetTitle>Datasets</SheetTitle>
          <SheetDescription>
            Survey simulation history and comparison.
          </SheetDescription>
        </SheetHeader>
        <div className="mt-6">
          <DatasetHistory />
        </div>
      </SheetContent>
    </Sheet>
  )
}
