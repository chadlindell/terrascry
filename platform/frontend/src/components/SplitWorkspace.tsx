/** Split view layout: 2D heatmap left, 3D scene right, with resizable divider. */

import { lazy, Suspense } from 'react'
import { Panel, Group, Separator } from 'react-resizable-panels'
import { useAppStore, type ViewMode } from '../stores/appStore'
import { useCrossSectionStore } from '../stores/crossSectionStore'
import { ErrorBoundary } from './ErrorBoundary'
import { LoadingSkeleton } from './LoadingSkeleton'
import { CrossSectionView } from './CrossSectionView'

const MapView = lazy(() => import('./MapView'))
const SceneView = lazy(() => import('./SceneView'))
const ComparisonView = lazy(() => import('./ComparisonView'))

const VIEW_MODES: { mode: ViewMode; label: string }[] = [
  { mode: '2d', label: '2D' },
  { mode: 'split', label: 'Split' },
  { mode: '3d', label: '3D' },
  { mode: 'comparison', label: 'Compare' },
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

function MapPanel() {
  return (
    <ErrorBoundary>
      <Suspense fallback={<LoadingSkeleton />}>
        <MapView />
      </Suspense>
    </ErrorBoundary>
  )
}

function ScenePanel() {
  return (
    <ErrorBoundary>
      <Suspense fallback={<LoadingSkeleton />}>
        <SceneView />
      </Suspense>
    </ErrorBoundary>
  )
}

function WithCrossSection({ children }: { children: React.ReactNode }) {
  const hasProfile = useCrossSectionStore((s) => s.profileData.length > 0)

  if (!hasProfile) {
    return <>{children}</>
  }

  return (
    <Group direction="vertical" className="h-full">
      <Panel defaultSize={70} minSize={30}>
        <div className="h-full">{children}</div>
      </Panel>
      <Separator className="h-1 bg-zinc-700 hover:bg-emerald-500 transition-colors cursor-row-resize" />
      <Panel defaultSize={30} minSize={15}>
        <CrossSectionView />
      </Panel>
    </Group>
  )
}

export function SplitWorkspace() {
  const viewMode = useAppStore((s) => s.viewMode)

  return (
    <div className="relative w-full h-full">
      <ViewModeControl />

      {viewMode === 'comparison' ? (
        <ErrorBoundary>
          <Suspense fallback={<LoadingSkeleton />}>
            <ComparisonView />
          </Suspense>
        </ErrorBoundary>
      ) : viewMode === 'split' ? (
        <WithCrossSection>
          <Group direction="horizontal" className="h-full">
            <Panel defaultSize={50} minSize={20}>
              <div className="h-full">
                <MapPanel />
              </div>
            </Panel>

            <Separator className="w-1 bg-zinc-700 hover:bg-emerald-500 transition-colors cursor-col-resize" />

            <Panel defaultSize={50} minSize={20}>
              <div className="h-full">
                <ScenePanel />
              </div>
            </Panel>
          </Group>
        </WithCrossSection>
      ) : viewMode === '2d' ? (
        <WithCrossSection>
          <MapPanel />
        </WithCrossSection>
      ) : (
        <ScenePanel />
      )}
    </div>
  )
}
