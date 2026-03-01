/** Split view layout: 2D heatmap left, 3D scene right, with resizable divider. */

import { lazy, Suspense } from 'react'
import { Panel, Group, Separator } from 'react-resizable-panels'
import { useAppStore } from '../stores/appStore'
import { useCrossSectionStore } from '../stores/crossSectionStore'
import { ErrorBoundary } from './ErrorBoundary'
import { LoadingSkeleton } from './LoadingSkeleton'
import { CrossSectionView } from './CrossSectionView'

const MapView = lazy(() => import('./MapView'))
const SceneView = lazy(() => import('./SceneView'))
const ComparisonView = lazy(() => import('./ComparisonView'))

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
      <Separator className="group relative h-1 bg-zinc-300 hover:bg-emerald-500 transition-colors cursor-row-resize">
        <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
          <span className="w-1 h-1 rounded-full bg-white" />
          <span className="w-1 h-1 rounded-full bg-white" />
          <span className="w-1 h-1 rounded-full bg-white" />
        </div>
      </Separator>
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
      {viewMode === 'comparison' ? (
        <div className="w-full h-full">
          <ErrorBoundary>
            <Suspense fallback={<LoadingSkeleton />}>
              <ComparisonView />
            </Suspense>
          </ErrorBoundary>
        </div>
      ) : viewMode === 'split' ? (
        <div className="w-full h-full">
          <WithCrossSection>
            <Group direction="horizontal" className="h-full">
              <Panel defaultSize={50} minSize={20}>
                <div className="h-full">
                  <MapPanel />
                </div>
              </Panel>

              <Separator className="group relative w-1 bg-zinc-300 hover:bg-emerald-500 transition-colors cursor-col-resize">
                <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 flex flex-col gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                  <span className="w-1 h-1 rounded-full bg-white" />
                  <span className="w-1 h-1 rounded-full bg-white" />
                  <span className="w-1 h-1 rounded-full bg-white" />
                </div>
              </Separator>

              <Panel defaultSize={50} minSize={20}>
                <div className="h-full">
                  <ScenePanel />
                </div>
              </Panel>
            </Group>
          </WithCrossSection>
        </div>
      ) : viewMode === '2d' ? (
        <div className="w-full h-full">
          <WithCrossSection>
            <MapPanel />
          </WithCrossSection>
        </div>
      ) : (
        <div className="w-full h-full">
          <ScenePanel />
        </div>
      )}
    </div>
  )
}
