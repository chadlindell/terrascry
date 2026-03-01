/** 3D scene view using React Three Fiber. */

import { useMemo, useEffect, useRef, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { useQueryClient } from '@tanstack/react-query'
import { useAppStore } from '../stores/appStore'
import { useAnomalies } from '../hooks/useAnomalies'
import { useColorScaleStore } from '../stores/colorScaleStore'
import { useScenarioDetail } from '../hooks/useScenarios'
import { TerrainMesh } from './TerrainMesh'
import { BuriedObjects } from './BuriedObjects'
import { SurveyPath3D } from './SurveyPath3D'
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

/** Scale markers rendered as simple Three.js line segments (no drei Text/Line to reduce GPU load). */
function SimpleScaleMarkers({ xMin, yMin, xMax, yMax, z }: {
  xMin: number; yMin: number; xMax: number; yMax: number; z: number
}) {
  const xRange = xMax - xMin
  const tickSpacing = xRange > 30 ? 10 : xRange > 10 ? 5 : xRange > 5 ? 2 : 1
  const offset = 0.3

  const positions = useMemo(() => {
    const pts: number[] = []
    // X-axis ticks along bottom edge
    const xStart = Math.ceil(xMin / tickSpacing) * tickSpacing
    for (let x = xStart; x <= xMax; x += tickSpacing) {
      pts.push(x, yMin - offset, z, x, yMin - offset - 0.3, z)
    }
    // Y-axis ticks along left edge
    const yStart = Math.ceil(yMin / tickSpacing) * tickSpacing
    for (let y = yStart; y <= yMax; y += tickSpacing) {
      pts.push(xMin - offset, y, z, xMin - offset - 0.3, y, z)
    }
    // Baselines
    pts.push(xMin, yMin - offset, z, xMax, yMin - offset, z)
    pts.push(xMin - offset, yMin, z, xMin - offset, yMax, z)
    return new Float32Array(pts)
  }, [xMin, yMin, xMax, yMax, z, tickSpacing, offset])

  return (
    <lineSegments>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <lineBasicMaterial color="#71717a" />
    </lineSegments>
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

  // Track WebGL context loss and recover
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [contextLost, setContextLost] = useState(false)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const onLost = (e: Event) => {
      e.preventDefault()
      setContextLost(true)
    }
    const onRestored = () => setContextLost(false)
    canvas.addEventListener('webglcontextlost', onLost)
    canvas.addEventListener('webglcontextrestored', onRestored)
    return () => {
      canvas.removeEventListener('webglcontextlost', onLost)
      canvas.removeEventListener('webglcontextrestored', onRestored)
    }
  }, [])

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

  if (contextLost) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3">
        <p className="text-sm font-medium text-zinc-500">WebGL context lost</p>
        <button
          className="text-xs text-emerald-600 underline"
          onClick={() => { setContextLost(false); window.location.reload() }}
        >
          Reload page
        </button>
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
      frameloop="always"
      gl={{ antialias: true, powerPreference: 'default' }}
      onCreated={({ gl }) => { canvasRef.current = gl.domElement }}
    >
      {/* Background + atmospheric fog */}
      <color attach="background" args={['#eef2f7']} />
      <fog attach="fog" args={['#eef2f7', 80, 150]} />

      {/* Lightweight lighting (no Environment HDR or shadow maps) */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[20, 20, 30]} intensity={1.0} />
      <directionalLight position={[-15, -10, 20]} intensity={0.3} color="#a1c4fd" />
      <hemisphereLight args={['#87ceeb', '#4a7c59', 0.3]} />

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
        <SimpleScaleMarkers {...terrainBounds} />
      )}
    </Canvas>
  )
}

export default SceneView
