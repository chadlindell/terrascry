/** Import data dialog wrapping the existing ImportPanel. */

import { useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogDescription,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Upload } from 'lucide-react'
import { ImportPanel } from '../ImportPanel'

export function ImportDialog() {
  const [open, setOpen] = useState(false)

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="sm" className="gap-1.5">
          <Upload className="h-3.5 w-3.5" />
          Import
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[480px]">
        <DialogHeader>
          <DialogTitle>Import Data</DialogTitle>
          <DialogDescription>
            Upload a CSV file with survey point data.
          </DialogDescription>
        </DialogHeader>
        <ImportPanel />
      </DialogContent>
    </Dialog>
  )
}
