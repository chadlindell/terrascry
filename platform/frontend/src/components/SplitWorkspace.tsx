/** Split view layout: 2D heatmap left, 3D scene right, with resizable divider. */

import { lazy, Suspense } from 'react'
import { Panel, Group, Separator } from 'react-resizable-panels'
import { useAppStore, type ViewMode } from '../stores/appStore'
import { ErrorBoundary } from './ErrorBoundary'
import { LoadingSkeleton } from './LoadingSkeleton'

const MapView = lazy(() => import('./MapView'))
const SceneView = lazy(() => import('./SceneView'))

const VIEW_MODES: { mode: ViewMode; label: string }[] = [
  { mode: '2d', label: '2D' },
  { mode: 'split', label: 'Split' },
  { mode: '3d', label: '3D' },
]

function ViewModeControl() {
  const viewMode = useAppStore((s) => s.viewMode)
  const setViewMode = useAppStore((s) => s.setViewMode)

  return (
    <div className="absolute top-3 right-3 z-10 flex rounded-md bg-zinc-800/80 backdrop-blur-sm border border-zinc-700/50 overflow-hidden">
      {VIEW_MODES.map(({ mode, label }) => (
        <button
          key={mode}
          onClick={() => setViewMode(mode)}
          className={`px-3 py-1 text-xs font-medium transition-colors ${
            viewMode === mode
              ? 'bg-emerald-600 text-white'
              : 'text-zinc-400 hover:text-zinc-100 hover:bg-zinc-700/50'
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  )
}

export function SplitWorkspace() {
  const viewMode = useAppStore((s) => s.viewMode)

  return (
    <div className="relative w-full h-full">
      <ViewModeControl />

      {viewMode === 'split' ? (
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
            <div className="h-full">
              <ErrorBoundary>
                <Suspense fallback={<LoadingSkeleton />}>
                  <SceneView />
                </Suspense>
              </ErrorBoundary>
            </div>
          </Panel>
        </Group>
      ) : viewMode === '2d' ? (
        <ErrorBoundary>
          <Suspense fallback={<LoadingSkeleton />}>
            <MapView />
          </Suspense>
        </ErrorBoundary>
      ) : (
        <ErrorBoundary>
          <Suspense fallback={<LoadingSkeleton />}>
            <SceneView />
          </Suspense>
        </ErrorBoundary>
      )}
    </div>
  )
}
