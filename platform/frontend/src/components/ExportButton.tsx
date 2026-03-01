/** Export dropdown button — download dataset as CSV, Grid CSV, or ESRI ASC. */

import { useState, useRef, useEffect } from 'react'
import { useAppStore } from '../stores/appStore'

const FORMATS = [
  { value: 'csv', label: 'CSV (survey points)' },
  { value: 'grid_csv', label: 'Grid CSV' },
  { value: 'asc', label: 'ESRI ASCII Grid' },
] as const

export function ExportButton() {
  const activeDatasetId = useAppStore((s) => s.activeDatasetId)
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  // Close dropdown on outside click
  useEffect(() => {
    if (!open) return
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [open])

  if (!activeDatasetId) return null

  const handleExport = (format: string) => {
    window.open(`/api/datasets/${activeDatasetId}/export?format=${format}`, '_blank')
    setOpen(false)
  }

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded text-sm font-medium transition-colors bg-zinc-200 hover:bg-zinc-300 text-zinc-700"
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
        </svg>
        Export
      </button>

      {open && (
        <div className="absolute bottom-full left-0 right-0 mb-1 rounded-lg glass-panel-strong shadow-overlay overflow-hidden z-20">
          {FORMATS.map(({ value, label }) => (
            <button
              key={value}
              onClick={() => handleExport(value)}
              className="w-full text-left px-3 py-2 text-sm text-zinc-700 hover:bg-zinc-100 transition-colors"
            >
              {label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
