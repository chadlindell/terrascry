/** Flat terrain plane matching scenario extents with optional heatmap texture. */

import { useEffect, useMemo, useRef } from 'react'
import { Grid } from '@react-three/drei'
import * as THREE from 'three'
import type { ScenarioDetail, Dataset } from '../api'
import type { ColormapName } from '../colormap'
import { gridToImageData } from '../colormap'

interface TerrainMeshProps {
  scenario: ScenarioDetail
  dataset?: Dataset
  colormap?: ColormapName
  range?: [number, number]
}

export function TerrainMesh({ scenario, dataset, colormap, range }: TerrainMeshProps) {
  const { terrain } = scenario
  const xSize = terrain.x_extent[1] - terrain.x_extent[0]
  const ySize = terrain.y_extent[1] - terrain.y_extent[0]
  const cx = (terrain.x_extent[0] + terrain.x_extent[1]) / 2
  const cy = (terrain.y_extent[0] + terrain.y_extent[1]) / 2
  const z = terrain.surface_elevation

  const position = useMemo(
    () => [cx, cy, z] as [number, number, number],
    [cx, cy, z],
  )

  const textureRef = useRef<THREE.CanvasTexture | null>(null)

  // Generate heatmap texture from grid data
  const texture = useMemo(() => {
    if (!dataset || !colormap || !range) return null

    const imageData = gridToImageData(dataset.grid_data, colormap, range)

    // Draw ImageData onto an offscreen canvas
    const canvas = document.createElement('canvas')
    canvas.width = imageData.width
    canvas.height = imageData.height
    const ctx = canvas.getContext('2d')
    if (!ctx) return null
    ctx.putImageData(imageData, 0, 0)

    const tex = new THREE.CanvasTexture(canvas)
    tex.minFilter = THREE.LinearFilter
    tex.magFilter = THREE.LinearFilter
    return tex
  }, [dataset, colormap, range])

  // Dispose previous texture when a new one is created
  useEffect(() => {
    const prev = textureRef.current
    textureRef.current = texture
    return () => {
      prev?.dispose()
    }
  }, [texture])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      textureRef.current?.dispose()
    }
  }, [])

  return (
    <group>
      {/* Ground plane */}
      <mesh position={position} receiveShadow>
        <planeGeometry args={[xSize, ySize]} />
        {texture ? (
          <meshStandardMaterial map={texture} />
        ) : (
          <meshStandardMaterial color="#3f3f46" transparent opacity={0.4} />
        )}
      </mesh>

      {/* Grid helper */}
      <Grid
        position={[cx, cy, z + 0.001]}
        args={[xSize, ySize]}
        cellSize={1}
        cellColor="#52525b"
        sectionSize={5}
        sectionColor="#71717a"
        fadeDistance={50}
        infiniteGrid={false}
      />
    </group>
  )
}
