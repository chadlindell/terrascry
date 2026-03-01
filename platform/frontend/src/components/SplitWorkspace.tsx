/** Split view layout: 2D heatmap left, 3D scene right, with resizable divider. */

import { lazy, Suspense } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
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

const panelVariants = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  exit: { opacity: 0 },
}

export function SplitWorkspace() {
  const viewMode = useAppStore((s) => s.viewMode)

  return (
    <div className="relative w-full h-full">
      <AnimatePresence mode="wait">
        {viewMode === 'comparison' ? (
          <motion.div key="comparison" className="w-full h-full" variants={panelVariants} initial="initial" animate="animate" exit="exit" transition={{ duration: 0.2 }}>
            <ErrorBoundary>
              <Suspense fallback={<LoadingSkeleton />}>
                <ComparisonView />
              </Suspense>
            </ErrorBoundary>
          </motion.div>
        ) : viewMode === 'split' ? (
          <motion.div key="split" className="w-full h-full" variants={panelVariants} initial="initial" animate="animate" exit="exit" transition={{ duration: 0.2 }}>
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
          </motion.div>
        ) : viewMode === '2d' ? (
          <motion.div key="2d" className="w-full h-full" variants={panelVariants} initial="initial" animate="animate" exit="exit" transition={{ duration: 0.2 }}>
            <WithCrossSection>
              <MapPanel />
            </WithCrossSection>
          </motion.div>
        ) : (
          <motion.div key="3d" className="w-full h-full" variants={panelVariants} initial="initial" animate="animate" exit="exit" transition={{ duration: 0.2 }}>
            <ScenePanel />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
