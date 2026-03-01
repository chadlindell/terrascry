/** 3D scene view using React Three Fiber with post-processing, annotations, and environment. */

import { useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Environment } from '@react-three/drei'
import { EffectComposer, SSAO, Bloom, Vignette } from '@react-three/postprocessing'
import { useQueryClient } from '@tanstack/react-query'
import { useAppStore } from '../stores/appStore'
import { useAnomalies } from '../hooks/useAnomalies'
import { useColorScaleStore } from '../stores/colorScaleStore'
import { useScenarioDetail } from '../hooks/useScenarios'
import { TerrainMesh } from './TerrainMesh'
import { BuriedObjects } from './BuriedObjects'
import { SurveyPath3D } from './SurveyPath3D'
import { AxisWidget, ScaleMarkers3D } from './SceneAnnotations'
import type { Dataset, AnomalyCell } from '../api'

function AnomalyMarkers3D({ anomalies }: { anomalies: AnomalyCell[] }) {
  return (
    <>
      {anomalies.map((a, i) => (
        <mesh key={i} position={[a.x, a.y, 0.1]}>
          <sphereGeometry args={[0.3, 16, 16]} />
          <meshStandardMaterial
            color="#ff3c3c"
            transparent
            opacity={0.4}
            depthWrite={false}
            emissive="#ff3c3c"
            emissiveIntensity={2.0}
          />
        </mesh>
      ))}
    </>
  )
}

export function SceneView() {
  const selectedScenario = useAppStore((s) => s.selectedScenario)
  const activeDatasetId = useAppStore((s) => s.activeDatasetId)
  const showAnomalies = useAppStore((s) => s.showAnomalies)
  const queryClient = useQueryClient()
  const colormap = useColorScaleStore((s) => s.colormap)
  const rangeMin = useColorScaleStore((s) => s.rangeMin)
  const rangeMax = useColorScaleStore((s) => s.rangeMax)

  const { data: scenario } = useScenarioDetail(selectedScenario)

  const dataset = activeDatasetId
    ? queryClient.getQueryData<Dataset>(['dataset', activeDatasetId])
    : null

  const { data: fetchedAnomalies } = useAnomalies(
    showAnomalies ? activeDatasetId : null
  )
  const anomalyCells: AnomalyCell[] = fetchedAnomalies ?? []

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

  // Terrain bounds for scale markers
  const terrainBounds = useMemo(() => {
    if (!scenario) return null
    const t = scenario.terrain
    return {
      xMin: t.x_extent[0],
      yMin: t.y_extent[0],
      xMax: t.x_extent[1],
      yMax: t.y_extent[1],
      z: t.surface_elevation,
    }
  }, [scenario])

  if (!scenario) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3">
        <div className="w-12 h-12 rounded-full bg-zinc-100 flex items-center justify-center">
          <svg className="w-6 h-6 text-zinc-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 7.5l-2.25-1.313M21 7.5v2.25m0-2.25l-2.25 1.313M3 7.5l2.25-1.313M3 7.5l2.25 1.313M3 7.5v2.25m9 3l2.25-1.313M12 12.75l-2.25-1.313M12 12.75V15m0 6.75l2.25-1.313M12 21.75V19.5m0 2.25l-2.25-1.313m0-16.875L12 2.25l2.25 1.313M21 14.25v2.25l-2.25 1.313m-13.5 0L3 16.5v-2.25" />
          </svg>
        </div>
        <div className="text-center">
          <p className="text-sm font-medium text-zinc-500">No 3D scene</p>
          <p className="text-xs text-zinc-400 mt-0.5">Select a scenario to view the 3D scene</p>
        </div>
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
      shadows
    >
      {/* Background + atmospheric fog */}
      <color attach="background" args={['#eef2f7']} />
      <fog attach="fog" args={['#eef2f7', 80, 150]} />

      {/* PBR environment for subtle reflections */}
      <Environment preset="city" />

      {/* Rich lighting rig */}
      <ambientLight intensity={0.25} />
      <directionalLight
        position={[20, 20, 30]}
        intensity={0.8}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      <directionalLight position={[-15, -10, 20]} intensity={0.15} color="#a1c4fd" />
      <hemisphereLight args={['#87ceeb', '#4a7c59', 0.2]} />

      {/* Controls */}
      <OrbitControls target={cameraTarget} makeDefault />

      {/* Terrain with elevation relief */}
      <TerrainMesh
        scenario={scenario}
        dataset={dataset ?? undefined}
        colormap={colormap}
        range={[rangeMin, rangeMax]}
        verticalExaggeration={5}
      />

      {/* Buried objects with labels */}
      {scenario.objects.length > 0 && (
        <BuriedObjects objects={scenario.objects} />
      )}

      {/* Survey path */}
      {dataset && dataset.survey_points.length > 1 && (
        <SurveyPath3D points={dataset.survey_points} />
      )}

      {/* Anomaly markers */}
      {showAnomalies && anomalyCells.length > 0 && (
        <AnomalyMarkers3D anomalies={anomalyCells} />
      )}

      {/* Scale markers along terrain edges */}
      {terrainBounds && (
        <ScaleMarkers3D {...terrainBounds} />
      )}

      {/* Axis widget in corner */}
      <AxisWidget />

      {/* Post-processing effects */}
      <EffectComposer>
        <SSAO radius={0.4} intensity={30} luminanceInfluence={0.6} />
        <Bloom luminanceThreshold={0.9} intensity={0.3} mipmapBlur />
        <Vignette offset={0.2} darkness={0.3} />
      </EffectComposer>
    </Canvas>
  )
}

export default SceneView
