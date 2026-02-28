/** Scrollable list of past simulation datasets with selection, delete, compare, and virtual scrolling. */

import { useRef } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import { useAppStore } from '../stores/appStore'
import { useComparisonStore } from '../stores/comparisonStore'
import { useDatasets, useDeleteDataset } from '../hooks/useDatasets'

export function DatasetHistory() {
  const activeDatasetId = useAppStore((s) => s.activeDatasetId)
  const setActiveDatasetId = useAppStore((s) => s.setActiveDatasetId)
  const setViewMode = useAppStore((s) => s.setViewMode)
  const setComparisonDatasetId = useComparisonStore((s) => s.setComparisonDatasetId)
  const { data: datasets, isLoading } = useDatasets()
  const { mutate: deleteMutation } = useDeleteDataset()
  const scrollRef = useRef<HTMLDivElement>(null)

  const rowVirtualizer = useVirtualizer({
    count: datasets?.length ?? 0,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 56,
    overscan: 5,
  })

  const handleDelete = (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    deleteMutation(id)
    if (activeDatasetId === id) {
      setActiveDatasetId(null)
    }
  }

  const handleCompare = (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    setComparisonDatasetId(id)
    setViewMode('comparison')
  }

  if (isLoading) {
    return (
      <div className="px-4 py-3">
        <p className="text-xs text-zinc-400">Loading datasets...</p>
      </div>
    )
  }

  if (!datasets || datasets.length === 0) {
    return (
      <div className="px-4 py-3">
        <p className="text-xs text-zinc-400">
          No datasets yet. Run a survey or import data to get started.
        </p>
      </div>
    )
  }

  return (
    <div className="px-4 py-3">
      <h2 className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2">
        History
      </h2>
      <div
        ref={scrollRef}
        className="overflow-y-auto"
        style={{ maxHeight: 300 }}
      >
        <div
          style={{
            height: `${rowVirtualizer.getTotalSize()}px`,
            width: '100%',
            position: 'relative',
          }}
        >
          {rowVirtualizer.getVirtualItems().map((virtualRow) => {
            const meta = datasets[virtualRow.index]
            const isActive = meta.id === activeDatasetId
            const time = new Date(meta.created_at).toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            })
            const date = new Date(meta.created_at).toLocaleDateString([], {
              month: 'short',
              day: 'numeric',
            })

            return (
              <div
                key={meta.id}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: `${virtualRow.size}px`,
                  transform: `translateY(${virtualRow.start}px)`,
                }}
              >
                <button
                  onClick={() => setActiveDatasetId(meta.id)}
                  className={`w-full flex items-center justify-between gap-2 px-3 py-2 rounded text-left text-sm transition-colors group ${
                    isActive
                      ? 'bg-emerald-50 text-emerald-700 border border-emerald-300'
                      : 'text-zinc-700 hover:bg-white border border-transparent'
                  }`}
                >
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-medium">{meta.scenario_name}</p>
                    <p className="text-xs text-zinc-400">
                      {date} {time}
                    </p>
                  </div>
                  <div className="shrink-0 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={(e) => handleCompare(e, meta.id)}
                      className="p-1 rounded text-zinc-400 hover:text-blue-500 hover:bg-zinc-200/50"
                      aria-label={`Compare ${meta.scenario_name}`}
                      title="Compare"
                    >
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM21 16c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2z" />
                      </svg>
                    </button>
                    <button
                      onClick={(e) => handleDelete(e, meta.id)}
                      className="p-1 rounded text-zinc-400 hover:text-red-500 hover:bg-zinc-200/50"
                      aria-label={`Delete ${meta.scenario_name}`}
                    >
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                </button>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
