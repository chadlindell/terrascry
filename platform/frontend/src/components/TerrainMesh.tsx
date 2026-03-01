/** Terrain mesh with heatmap texture and optional elevation relief from gradient values. */

import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import type { ScenarioDetail, Dataset } from '../api'
import type { ColormapName } from '../colormap'
import { gridToImageData, getValueRange } from '../colormap'

interface TerrainMeshProps {
  scenario: ScenarioDetail
  dataset?: Dataset
  colormap?: ColormapName
  range?: [number, number]
  verticalExaggeration?: number
}

export function TerrainMesh({ scenario, dataset, colormap, range, verticalExaggeration = 5 }: TerrainMeshProps) {
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
  const geometryRef = useRef<THREE.PlaneGeometry | null>(null)

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

  // Compute geometry segments from grid resolution
  const segments = useMemo(() => {
    if (!dataset) return { xSegs: 1, ySegs: 1 }
    return {
      xSegs: Math.max(1, dataset.grid_data.cols - 1),
      ySegs: Math.max(1, dataset.grid_data.rows - 1),
    }
  }, [dataset])

  // Apply vertex displacement based on gradient values
  useEffect(() => {
    const geom = geometryRef.current
    if (!geom || !dataset) return

    const { values, rows, cols } = dataset.grid_data
    const posAttr = geom.attributes.position
    if (!posAttr) return

    const [dataMin, dataMax] = getValueRange(values)
    const dataSpan = dataMax - dataMin || 1

    // The PlaneGeometry has (xSegs+1) * (ySegs+1) vertices
    // Vertices are laid out row by row: y varies from +ySize/2 to -ySize/2 (top to bottom)
    // x varies from -xSize/2 to +xSize/2 (left to right)
    const xVerts = segments.xSegs + 1
    const yVerts = segments.ySegs + 1

    for (let iy = 0; iy < yVerts; iy++) {
      for (let ix = 0; ix < xVerts; ix++) {
        const vertIdx = iy * xVerts + ix
        // Map vertex grid position to data grid position
        const dataCol = Math.min(cols - 1, Math.round((ix / segments.xSegs) * (cols - 1)))
        // PlaneGeometry rows go top-to-bottom, our grid is bottom-to-top, so flip
        const dataRow = Math.min(rows - 1, Math.round(((yVerts - 1 - iy) / segments.ySegs) * (rows - 1)))
        const dataIdx = dataRow * cols + dataCol
        const normalized = (values[dataIdx] - dataMin) / dataSpan
        // Displace Z vertex (which is the 3rd component in the position array)
        posAttr.setZ(vertIdx, normalized * verticalExaggeration)
      }
    }

    posAttr.needsUpdate = true
    geom.computeVertexNormals()
  }, [dataset, segments, verticalExaggeration])

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
    <mesh position={position}>
      <planeGeometry
        ref={geometryRef}
        args={[xSize, ySize, segments.xSegs, segments.ySegs]}
      />
      {texture ? (
        <meshStandardMaterial
          map={texture}
          roughness={0.8}
          metalness={0}
          side={THREE.DoubleSide}
        />
      ) : (
        <meshStandardMaterial
          color="#a1a1aa"
          roughness={0.8}
          metalness={0}
          side={THREE.DoubleSide}
        />
      )}
    </mesh>
  )
}
