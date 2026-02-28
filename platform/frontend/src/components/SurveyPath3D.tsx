/** Survey walk path rendered as a 3D line above terrain. */

import { useMemo } from 'react'
import { Line } from '@react-three/drei'
import type { SurveyPoint } from '../api'

interface SurveyPath3DProps {
  points: SurveyPoint[]
  height?: number
}

export function SurveyPath3D({ points, height = 0.05 }: SurveyPath3DProps) {
  const linePoints = useMemo(
    () => points.map((p) => [p.x, p.y, height] as [number, number, number]),
    [points, height],
  )

  if (linePoints.length < 2) return null

  return (
    <Line
      points={linePoints}
      color="#ffffff"
      lineWidth={1.5}
      opacity={0.7}
      transparent
    />
  )
}
