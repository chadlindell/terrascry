/** 3D scene annotations: axis widget and scale markers along terrain edges. */

import { Text, Line, Hud, OrthographicCamera } from '@react-three/drei'
import * as THREE from 'three'

// Pre-allocated vectors to avoid per-render allocations
const AXIS_X = new THREE.Vector3(1, 0, 0)
const AXIS_Y = new THREE.Vector3(0, 1, 0)
const AXIS_Z = new THREE.Vector3(0, 0, 1)
const ORIGIN = new THREE.Vector3(0, 0, 0)

/** Small XYZ axis widget in the corner of the 3D view. */
export function AxisWidget() {
  const length = 1
  const headLength = 0.2
  const headWidth = 0.08

  return (
    <Hud renderPriority={2}>
      <OrthographicCamera makeDefault position={[0, 0, 10]} zoom={80} />
      <ambientLight intensity={1} />
      <group position={[-3.2, -2.5, 0]}>
        {/* X axis (East) — red */}
        <arrowHelper args={[AXIS_X, ORIGIN, length, 0xef4444, headLength, headWidth]} />
        <Text position={[length + 0.2, 0, 0]} fontSize={0.25} color="#ef4444" anchorX="center" anchorY="middle" font={undefined}>
          E
        </Text>

        {/* Y axis (North) — green */}
        <arrowHelper args={[AXIS_Y, ORIGIN, length, 0x22c55e, headLength, headWidth]} />
        <Text position={[0, length + 0.2, 0]} fontSize={0.25} color="#22c55e" anchorX="center" anchorY="middle" font={undefined}>
          N
        </Text>

        {/* Z axis (Up) — blue */}
        <arrowHelper args={[AXIS_Z, ORIGIN, length, 0x3b82f6, headLength, headWidth]} />
        <Text position={[0, 0, length + 0.2]} fontSize={0.25} color="#3b82f6" anchorX="center" anchorY="middle" font={undefined}>
          Z
        </Text>
      </group>
    </Hud>
  )
}

/** Distance tick marks along terrain edges. */
interface ScaleMarkers3DProps {
  xMin: number
  yMin: number
  xMax: number
  yMax: number
  z: number
}

export function ScaleMarkers3D({ xMin, yMin, xMax, yMax, z }: ScaleMarkers3DProps) {
  const xRange = xMax - xMin
  const yRange = yMax - yMin

  // Pick a nice tick spacing
  const tickSpacing = xRange > 30 ? 10 : xRange > 10 ? 5 : xRange > 5 ? 2 : 1

  const xTicks: number[] = []
  const start = Math.ceil(xMin / tickSpacing) * tickSpacing
  for (let x = start; x <= xMax; x += tickSpacing) {
    xTicks.push(x)
  }

  const yTicks: number[] = []
  const yStart = Math.ceil(yMin / tickSpacing) * tickSpacing
  for (let y = yStart; y <= yMax; y += tickSpacing) {
    yTicks.push(y)
  }

  const tickLen = Math.max(xRange, yRange) * 0.015
  const offset = 0.3 // offset from terrain edge

  return (
    <group>
      {/* X-axis ticks along bottom edge */}
      {xTicks.map((x) => (
        <group key={`xt-${x}`}>
          <Line
            points={[[x, yMin - offset, z], [x, yMin - offset - tickLen, z]]}
            color="#71717a"
            lineWidth={1}
          />
          <Text
            position={[x, yMin - offset - tickLen - 0.3, z]}
            fontSize={0.3}
            color="#71717a"
            anchorX="center"
            anchorY="top"
            font={undefined}
          >
            {`${x.toFixed(0)}m`}
          </Text>
        </group>
      ))}

      {/* Y-axis ticks along left edge */}
      {yTicks.map((y) => (
        <group key={`yt-${y}`}>
          <Line
            points={[[xMin - offset, y, z], [xMin - offset - tickLen, y, z]]}
            color="#71717a"
            lineWidth={1}
          />
          <Text
            position={[xMin - offset - tickLen - 0.3, y, z]}
            fontSize={0.3}
            color="#71717a"
            anchorX="right"
            anchorY="middle"
            font={undefined}
          >
            {`${y.toFixed(0)}m`}
          </Text>
        </group>
      ))}

      {/* Axis baseline along bottom and left edges */}
      <Line
        points={[[xMin, yMin - offset, z], [xMax, yMin - offset, z]]}
        color="#a1a1aa"
        lineWidth={1}
      />
      <Line
        points={[[xMin - offset, yMin, z], [xMin - offset, yMax, z]]}
        color="#a1a1aa"
        lineWidth={1}
      />
    </group>
  )
}
