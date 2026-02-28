/** Import panel — drag-and-drop CSV upload with validation feedback. */

import { useState, useCallback } from 'react'
import { useImport } from '../hooks/useImport'

const MAX_FILE_SIZE = 10 * 1024 * 1024 // 10 MB

export function ImportPanel() {
  const [expanded, setExpanded] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const { mutate, isPending, error, isSuccess, reset } = useImport()

  const handleFile = useCallback(
    (file: File) => {
      reset()
      if (file.size > MAX_FILE_SIZE) {
        return
      }
      if (!file.name.endsWith('.csv')) {
        return
      }
      mutate(file)
    },
    [mutate, reset],
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragOver(false)
      const file = e.dataTransfer.files[0]
      if (file) handleFile(file)
    },
    [handleFile],
  )

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setDragOver(false)
  }, [])

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) handleFile(file)
      e.target.value = ''
    },
    [handleFile],
  )

  return (
    <div className="px-4 py-3">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1 text-xs font-medium text-zinc-400 uppercase tracking-wider hover:text-zinc-200 transition-colors"
      >
        <svg
          className={`w-3 h-3 transition-transform ${expanded ? 'rotate-90' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
        </svg>
        Import Data
      </button>

      {expanded && (
        <div className="mt-2">
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            className={`relative flex flex-col items-center justify-center gap-2 p-4 rounded border-2 border-dashed transition-colors cursor-pointer ${
              dragOver
                ? 'border-emerald-500 bg-emerald-500/10'
                : 'border-zinc-700 hover:border-zinc-600'
            }`}
          >
            {isPending ? (
              <div className="flex items-center gap-2">
                <svg className="animate-spin h-4 w-4 text-emerald-400" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                <span className="text-xs text-zinc-400">Uploading...</span>
              </div>
            ) : (
              <>
                <svg className="w-6 h-6 text-zinc-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                </svg>
                <p className="text-xs text-zinc-500">
                  Drop CSV file here or click to browse
                </p>
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleInputChange}
                  className="absolute inset-0 opacity-0 cursor-pointer"
                />
              </>
            )}
          </div>

          {isSuccess && (
            <p className="mt-2 text-xs text-emerald-400">
              File imported successfully
            </p>
          )}

          {error && (
            <p className="mt-2 text-xs text-red-400">
              {(error as Error).message}
            </p>
          )}
        </div>
      )}
    </div>
  )
}
