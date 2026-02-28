/** 3D scene view using React Three Fiber. */

import { useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { useQueryClient } from '@tanstack/react-query'
import { useAppStore } from '../stores/appStore'
import { useScenarioDetail } from '../hooks/useScenarios'
import { TerrainMesh } from './TerrainMesh'
import { BuriedObjects } from './BuriedObjects'
import { SurveyPath3D } from './SurveyPath3D'
import type { Dataset } from '../api'

export function SceneView() {
  const selectedScenario = useAppStore((s) => s.selectedScenario)
  const activeDatasetId = useAppStore((s) => s.activeDatasetId)
  const queryClient = useQueryClient()

  const { data: scenario } = useScenarioDetail(selectedScenario)

  const dataset = activeDatasetId
    ? queryClient.getQueryData<Dataset>(['dataset', activeDatasetId])
    : null

  // Compute camera target from scenario terrain
  const cameraTarget = useMemo(() => {
    if (!scenario) return [10, 10, 0] as [number, number, number]
    const t = scenario.terrain
    return [
      (t.x_extent[0] + t.x_extent[1]) / 2,
      (t.y_extent[0] + t.y_extent[1]) / 2,
      0,
    ] as [number, number, number]
  }, [scenario])

  const cameraPosition = useMemo(() => {
    return [
      cameraTarget[0] - 15,
      cameraTarget[1] - 15,
      20,
    ] as [number, number, number]
  }, [cameraTarget])

  if (!scenario) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-zinc-500 text-sm">
          Select a scenario to view 3D scene.
        </p>
      </div>
    )
  }

  return (
    <Canvas
      camera={{
        position: cameraPosition,
        up: [0, 0, 1],
        fov: 50,
        near: 0.1,
        far: 500,
      }}
      frameloop="demand"
      style={{ background: '#09090b' }}
    >
      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <directionalLight position={[20, 20, 30]} intensity={0.8} />

      {/* Controls */}
      <OrbitControls target={cameraTarget} makeDefault />

      {/* Terrain */}
      <TerrainMesh scenario={scenario} />

      {/* Buried objects */}
      {scenario.objects.length > 0 && (
        <BuriedObjects objects={scenario.objects} />
      )}

      {/* Survey path */}
      {dataset && dataset.survey_points.length > 1 && (
        <SurveyPath3D points={dataset.survey_points} />
      )}
    </Canvas>
  )
}
