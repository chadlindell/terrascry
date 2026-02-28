/** Dataset comparison view — side-by-side or difference mode. */

import { useMemo, lazy, Suspense } from 'react'
import { Panel, Group, Separator } from 'react-resizable-panels'
import { useQueryClient } from '@tanstack/react-query'
import { useAppStore } from '../stores/appStore'
import { useComparisonStore } from '../stores/comparisonStore'
import { useColorScaleStore } from '../stores/colorScaleStore'
import { useDatasets } from '../hooks/useDatasets'
import { computeDifference } from '../utils/gridDiff'
import { gridToImageData, getValueRange } from '../colormap'
import { ErrorBoundary } from './ErrorBoundary'
import { LoadingSkeleton } from './LoadingSkeleton'
import type { Dataset } from '../api'

const MapView = lazy(() => import('./MapView'))

function DifferenceView() {
  const activeDatasetId = useAppStore((s) => s.activeDatasetId)
  const comparisonDatasetId = useComparisonStore((s) => s.comparisonDatasetId)
  const queryClient = useQueryClient()

  const datasetA = activeDatasetId
    ? queryClient.getQueryData<Dataset>(['dataset', activeDatasetId])
    : null
  const datasetB = comparisonDatasetId
    ? queryClient.getQueryData<Dataset>(['dataset', comparisonDatasetId])
    : null

  const diffImageData = useMemo(() => {
    if (!datasetA || !datasetB) return null
    const diffGrid = computeDifference(datasetA.grid_data, datasetB.grid_data)
    const [min, max] = getValueRange(diffGrid.values)
    const absMax = Math.max(Math.abs(min), Math.abs(max)) || 1
    return {
      imageData: gridToImageData(diffGrid, 'diverging', [-absMax, absMax]),
      range: [-absMax, absMax] as [number, number],
      grid: diffGrid,
    }
  }, [datasetA, datasetB])

  if (!datasetA || !datasetB || !diffImageData) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-zinc-500 text-sm">
          Select two datasets to compare (active + comparison).
        </p>
      </div>
    )
  }

  const g = diffImageData.grid
  const [rMin, rMax] = diffImageData.range

  return (
    <div className="relative w-full h-full">
      <div className="absolute top-3 left-3 z-10 bg-zinc-800/80 backdrop-blur-sm border border-zinc-700/50 rounded p-2 text-xs text-zinc-300">
        <p>Difference: {datasetA.metadata.scenario_name} − {datasetB.metadata.scenario_name}</p>
        <p className="text-zinc-500">Range: {rMin.toFixed(1)} to {rMax.toFixed(1)} nT</p>
      </div>
      {/* Render difference as a simple canvas */}
      <DiffCanvas imageData={diffImageData.imageData} cols={g.cols} rows={g.rows} />
    </div>
  )
}

function DiffCanvas({ imageData, cols, rows }: { imageData: ImageData; cols: number; rows: number }) {
  const canvasRef = (canvas: HTMLCanvasElement | null) => {
    if (!canvas) return
    canvas.width = cols
    canvas.height = rows
    const ctx = canvas.getContext('2d')
    if (ctx) {
      ctx.putImageData(imageData, 0, 0)
    }
  }

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-full"
      style={{ imageRendering: 'pixelated', objectFit: 'contain' }}
    />
  )
}

function ComparisonControls() {
  const mode = useComparisonStore((s) => s.mode)
  const setMode = useComparisonStore((s) => s.setMode)
  const comparisonDatasetId = useComparisonStore((s) => s.comparisonDatasetId)
  const setComparisonDatasetId = useComparisonStore((s) => s.setComparisonDatasetId)
  const { data: datasets } = useDatasets()

  return (
    <div className="absolute bottom-3 left-3 z-10 bg-zinc-800/90 backdrop-blur-sm border border-zinc-700/50 rounded p-2 flex items-center gap-3">
      <select
        value={comparisonDatasetId ?? ''}
        onChange={(e) => setComparisonDatasetId(e.target.value || null)}
        className="px-2 py-1 rounded bg-zinc-700 border border-zinc-600 text-xs text-zinc-300"
      >
        <option value="">Compare with...</option>
        {datasets?.map((d) => (
          <option key={d.id} value={d.id}>
            {d.scenario_name} — {new Date(d.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </option>
        ))}
      </select>
      <div className="flex gap-1">
        <button
          onClick={() => setMode('side-by-side')}
          className={`px-2 py-0.5 rounded text-[10px] font-medium ${
            mode === 'side-by-side' ? 'bg-emerald-600 text-white' : 'bg-zinc-700 text-zinc-400'
          }`}
        >
          Side-by-Side
        </button>
        <button
          onClick={() => setMode('difference')}
          className={`px-2 py-0.5 rounded text-[10px] font-medium ${
            mode === 'difference' ? 'bg-emerald-600 text-white' : 'bg-zinc-700 text-zinc-400'
          }`}
        >
          Difference
        </button>
      </div>
    </div>
  )
}

export function ComparisonView() {
  const mode = useComparisonStore((s) => s.mode)
  const comparisonDatasetId = useComparisonStore((s) => s.comparisonDatasetId)

  return (
    <div className="relative w-full h-full">
      <ComparisonControls />

      {!comparisonDatasetId ? (
        <div className="flex items-center justify-center h-full">
          <p className="text-zinc-500 text-sm">
            Select a comparison dataset below.
          </p>
        </div>
      ) : mode === 'side-by-side' ? (
        <Group direction="horizontal" className="h-full">
          <Panel defaultSize={50} minSize={20}>
            <div className="h-full">
              <ErrorBoundary>
                <Suspense fallback={<LoadingSkeleton />}>
                  <MapView />
                </Suspense>
              </ErrorBoundary>
            </div>
          </Panel>
          <Separator className="w-1 bg-zinc-700 hover:bg-emerald-500 transition-colors cursor-col-resize" />
          <Panel defaultSize={50} minSize={20}>
            <div className="h-full relative">
              <div className="absolute top-3 left-3 z-10 bg-zinc-800/80 backdrop-blur-sm border border-zinc-700/50 rounded px-2 py-1 text-xs text-zinc-400">
                Comparison dataset
              </div>
              <ErrorBoundary>
                <Suspense fallback={<LoadingSkeleton />}>
                  <MapView />
                </Suspense>
              </ErrorBoundary>
            </div>
          </Panel>
        </Group>
      ) : (
        <DifferenceView />
      )}
    </div>
  )
}

export default ComparisonView
