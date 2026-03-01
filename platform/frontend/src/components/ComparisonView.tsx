/** Dataset comparison view — side-by-side or difference mode with contour overlay. */

import { useMemo, lazy, Suspense, useCallback, useRef, useEffect } from 'react'
import { Panel, Group, Separator } from 'react-resizable-panels'
import { useQueryClient } from '@tanstack/react-query'
import { useAppStore } from '../stores/appStore'
import { useComparisonStore } from '../stores/comparisonStore'
import { useColorScaleStore } from '../stores/colorScaleStore'
import { useDatasets } from '../hooks/useDatasets'
import { computeDifference } from '../utils/gridDiff'
import { gridToImageData, getValueRange } from '../colormap'
import { generateContours } from '../utils/contours'
import { ErrorBoundary } from './ErrorBoundary'
import { LoadingSkeleton } from './LoadingSkeleton'
import type { Dataset, GridData } from '../api'

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

  const diffResult = useMemo(() => {
    if (!datasetA || !datasetB) return null
    const diffGrid = computeDifference(datasetA.grid_data, datasetB.grid_data)
    const [min, max] = getValueRange(diffGrid.values)
    const absMax = Math.max(Math.abs(min), Math.abs(max)) || 1
    return {
      imageData: gridToImageData(diffGrid, 'diverging', [-absMax, absMax]),
      range: [-absMax, absMax] as [number, number],
      grid: diffGrid,
      contours: generateContours(diffGrid, 10),
    }
  }, [datasetA, datasetB])

  if (!datasetA || !datasetB || !diffResult) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-zinc-400 text-sm">
          Select two datasets to compare (active + comparison).
        </p>
      </div>
    )
  }

  const g = diffResult.grid
  const [rMin, rMax] = diffResult.range

  return (
    <div className="relative w-full h-full">
      <div className="absolute top-3 left-3 z-10 glass-panel shadow-card rounded-lg p-2 text-xs text-zinc-700">
        <p>Difference: {datasetA.metadata.scenario_name} - {datasetB.metadata.scenario_name}</p>
        <p className="text-zinc-400">Range: {rMin.toFixed(1)} to {rMax.toFixed(1)} nT</p>
      </div>
      <DiffCanvas
        imageData={diffResult.imageData}
        grid={g}
        contours={diffResult.contours}
      />
    </div>
  )
}

/** Anti-aliased diff canvas with 4x supersampling and contour SVG overlay. */
function DiffCanvas({
  imageData,
  grid,
  contours,
}: {
  imageData: ImageData
  grid: GridData
  contours: ReturnType<typeof generateContours>
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const setupCanvas = useCallback(
    (canvas: HTMLCanvasElement | null) => {
      if (!canvas) return
      // 4x supersampling for anti-aliased rendering
      const scale = 4
      canvas.width = grid.cols * scale
      canvas.height = grid.rows * scale
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      // Draw source image data at native res onto a temp canvas
      const srcCanvas = document.createElement('canvas')
      srcCanvas.width = grid.cols
      srcCanvas.height = grid.rows
      const srcCtx = srcCanvas.getContext('2d')
      if (!srcCtx) return
      srcCtx.putImageData(imageData, 0, 0)

      // Draw upscaled with high-quality smoothing
      ctx.imageSmoothingEnabled = true
      ctx.imageSmoothingQuality = 'high'
      ctx.drawImage(srcCanvas, 0, 0, canvas.width, canvas.height)
    },
    [imageData, grid.cols, grid.rows],
  )

  // SVG contour overlay paths
  const svgPaths = useMemo(() => {
    return contours.map((cp, i) => {
      const pathStrings = cp.coordinates.map((ring) => {
        const points = ring.map(([x, y]) => {
          // Convert world coords to fractional position (0-1)
          const fx = ((x - grid.x_min) / (grid.cols * grid.dx)) * 100
          const fy = (1 - (y - grid.y_min) / (grid.rows * grid.dy)) * 100
          return `${fx},${fy}`
        })
        return `M${points.join('L')}Z`
      })
      return (
        <path
          key={i}
          d={pathStrings.join(' ')}
          fill="none"
          stroke="rgba(0,0,0,0.3)"
          strokeWidth="0.3"
        />
      )
    })
  }, [contours, grid])

  return (
    <div ref={containerRef} className="relative w-full h-full">
      <canvas
        ref={(el) => {
          (canvasRef as React.MutableRefObject<HTMLCanvasElement | null>).current = el
          setupCanvas(el)
        }}
        className="w-full h-full"
        style={{ objectFit: 'contain' }}
      />
      {/* SVG contour overlay */}
      <svg
        className="absolute inset-0 w-full h-full pointer-events-none"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
      >
        {svgPaths}
      </svg>
    </div>
  )
}

function ComparisonControls() {
  const mode = useComparisonStore((s) => s.mode)
  const setMode = useComparisonStore((s) => s.setMode)
  const comparisonDatasetId = useComparisonStore((s) => s.comparisonDatasetId)
  const setComparisonDatasetId = useComparisonStore((s) => s.setComparisonDatasetId)
  const { data: datasets } = useDatasets()

  return (
    <div className="absolute bottom-3 left-3 z-10 glass-panel-strong shadow-overlay rounded-lg p-2 flex items-center gap-3">
      <select
        value={comparisonDatasetId ?? ''}
        onChange={(e) => setComparisonDatasetId(e.target.value || null)}
        className="px-2 py-1 rounded bg-zinc-100 border border-zinc-300 text-xs text-zinc-700"
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
            mode === 'side-by-side' ? 'bg-emerald-600 text-white' : 'bg-zinc-200 text-zinc-500'
          }`}
        >
          Side-by-Side
        </button>
        <button
          onClick={() => setMode('difference')}
          className={`px-2 py-0.5 rounded text-[10px] font-medium ${
            mode === 'difference' ? 'bg-emerald-600 text-white' : 'bg-zinc-200 text-zinc-500'
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
        <div className="flex flex-col items-center justify-center h-full gap-3">
          <div className="w-12 h-12 rounded-full bg-zinc-100 flex items-center justify-center">
            <svg className="w-6 h-6 text-zinc-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" />
            </svg>
          </div>
          <div className="text-center">
            <p className="text-sm font-medium text-zinc-500">No comparison selected</p>
            <p className="text-xs text-zinc-400 mt-0.5">Select a comparison dataset below</p>
          </div>
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
          <Separator className="w-1 bg-zinc-300 hover:bg-emerald-500 transition-colors cursor-col-resize" />
          <Panel defaultSize={50} minSize={20}>
            <div className="h-full relative">
              <div className="absolute top-3 left-3 z-10 glass-panel shadow-card rounded-lg px-2 py-1 text-xs text-zinc-500">
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
