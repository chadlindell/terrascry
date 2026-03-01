/** Render buried objects as semi-transparent colored spheres with floating labels. */

import { Html } from '@react-three/drei'
import type { ObjectSummary } from '../api'

interface BuriedObjectsProps {
  objects: ObjectSummary[]
}

const TYPE_MATERIALS: Record<string, {
  color: string
  emissive: string
  emissiveIntensity: number
  roughness: number
  metalness: number
}> = {
  ferrous: {
    color: '#ef4444',
    emissive: '#ef4444',
    emissiveIntensity: 0.1,
    roughness: 0.4,
    metalness: 0.3,
  },
  'non-ferrous': {
    color: '#3b82f6',
    emissive: '#3b82f6',
    emissiveIntensity: 0.1,
    roughness: 0.6,
    metalness: 0.1,
  },
  unknown: {
    color: '#a855f7',
    emissive: '#a855f7',
    emissiveIntensity: 0.1,
    roughness: 0.6,
    metalness: 0.1,
  },
}

const TYPE_LABELS: Record<string, string> = {
  ferrous: 'Ferrous',
  'non-ferrous': 'Non-ferrous',
  unknown: 'Unknown',
}

export function BuriedObjects({ objects }: BuriedObjectsProps) {
  return (
    <group>
      {objects.map((obj) => {
        const mat = TYPE_MATERIALS[obj.object_type] ?? TYPE_MATERIALS.unknown
        const [x, y, z] = obj.position
        const radius = obj.radius || 0.2
        const typeLabel = TYPE_LABELS[obj.object_type] ?? 'Unknown'

        return (
          <group key={obj.name}>
            <mesh position={[x, y, z]} castShadow>
              <sphereGeometry args={[radius, 24, 24]} />
              <meshStandardMaterial
                color={mat.color}
                transparent
                opacity={0.6}
                depthWrite={false}
                emissive={mat.emissive}
                emissiveIntensity={mat.emissiveIntensity}
                roughness={mat.roughness}
                metalness={mat.metalness}
              />
            </mesh>

            {/* Floating label above object */}
            <Html
              position={[x, y, z + radius + 0.5]}
              center
              distanceFactor={15}
              occlude={false}
            >
              <div className="glass-panel rounded-md px-2 py-1 shadow-card pointer-events-none whitespace-nowrap">
                <div className="text-[10px] font-semibold text-zinc-800">{obj.name}</div>
                <div className="text-[9px] text-zinc-500">
                  {typeLabel} &middot; {Math.abs(z).toFixed(1)}m deep
                </div>
              </div>
            </Html>
          </group>
        )
      })}
    </group>
  )
}
