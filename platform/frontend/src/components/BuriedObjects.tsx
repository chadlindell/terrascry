/** Render buried objects as semi-transparent colored spheres. */

import type { ObjectSummary } from '../api'

interface BuriedObjectsProps {
  objects: ObjectSummary[]
}

const TYPE_COLORS: Record<string, string> = {
  ferrous: '#ef4444',     // red
  'non-ferrous': '#3b82f6', // blue
  unknown: '#a855f7',     // purple
}

export function BuriedObjects({ objects }: BuriedObjectsProps) {
  return (
    <group>
      {objects.map((obj) => {
        const color = TYPE_COLORS[obj.object_type] ?? TYPE_COLORS.unknown
        const [x, y, z] = obj.position
        const radius = obj.radius || 0.2

        return (
          <mesh key={obj.name} position={[x, y, z]}>
            <sphereGeometry args={[radius, 24, 24]} />
            <meshStandardMaterial
              color={color}
              transparent
              opacity={0.6}
              depthWrite={false}
            />
          </mesh>
        )
      })}
    </group>
  )
}
