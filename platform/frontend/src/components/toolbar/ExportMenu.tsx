/** Export dropdown menu for the toolbar. */

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Button } from '@/components/ui/button'
import { Download, ChevronDown } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import { useAppStore } from '../../stores/appStore'
import type { Dataset } from '../../api'

function downloadBlob(content: string, filename: string, mime: string) {
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export function ExportMenu() {
  const activeDatasetId = useAppStore((s) => s.activeDatasetId)
  const queryClient = useQueryClient()

  const dataset = activeDatasetId
    ? queryClient.getQueryData<Dataset>(['dataset', activeDatasetId])
    : null

  const exportCSV = () => {
    if (!dataset) return
    const rows = dataset.survey_points.map(
      (p) => `${p.x},${p.y},${p.gradient_nt}`
    )
    downloadBlob('x,y,gradient_nt\n' + rows.join('\n'), `survey-${dataset.id}.csv`, 'text/csv')
  }

  const exportGridCSV = () => {
    if (!dataset) return
    const g = dataset.grid_data
    const header = `# ${g.cols}x${g.rows} grid, origin=(${g.x_min},${g.y_min}), spacing=(${g.dx},${g.dy})`
    const rows: string[] = []
    for (let r = 0; r < g.rows; r++) {
      rows.push(g.values.slice(r * g.cols, (r + 1) * g.cols).join(','))
    }
    downloadBlob(header + '\n' + rows.join('\n'), `grid-${dataset.id}.csv`, 'text/csv')
  }

  const exportESRI = () => {
    if (!dataset) return
    const g = dataset.grid_data
    const hdr = [
      `ncols ${g.cols}`,
      `nrows ${g.rows}`,
      `xllcorner ${g.x_min}`,
      `yllcorner ${g.y_min}`,
      `cellsize ${g.dx}`,
      `NODATA_value -9999`,
    ].join('\n')
    const rows: string[] = []
    for (let r = g.rows - 1; r >= 0; r--) {
      rows.push(g.values.slice(r * g.cols, (r + 1) * g.cols).join(' '))
    }
    downloadBlob(hdr + '\n' + rows.join('\n'), `grid-${dataset.id}.asc`, 'text/plain')
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className="gap-1.5" disabled={!dataset}>
          <Download className="h-3.5 w-3.5" />
          Export
          <ChevronDown className="h-3 w-3 opacity-50" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={exportCSV}>
          CSV (survey points)
        </DropdownMenuItem>
        <DropdownMenuItem onClick={exportGridCSV}>
          Grid CSV (interpolated)
        </DropdownMenuItem>
        <DropdownMenuItem onClick={exportESRI}>
          ESRI ASCII Grid
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
