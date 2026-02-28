/** Flat terrain plane matching scenario extents with grid overlay. */

import { useMemo } from 'react'
import { Grid } from '@react-three/drei'
import type { ScenarioDetail } from '../api'

interface TerrainMeshProps {
  scenario: ScenarioDetail
}

export function TerrainMesh({ scenario }: TerrainMeshProps) {
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

  return (
    <group>
      {/* Ground plane */}
      <mesh position={position} receiveShadow>
        <planeGeometry args={[xSize, ySize]} />
        <meshStandardMaterial color="#3f3f46" transparent opacity={0.4} />
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
