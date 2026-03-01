/** Generate contour lines from grid data using d3-contour. */

import { contours as d3Contours } from 'd3-contour'
import type { GridData } from '../api'

export interface ContourPath {
  /** Contour level value */
  value: number
  /** Array of coordinate rings: each ring is [x, y][] in world coordinates */
  coordinates: [number, number][][]
  /** Best position for a text label (midpoint of longest ring) */
  labelPosition: [number, number]
  /** Tangent angle at the label position (degrees) */
  labelAngle: number
}

/**
 * Find the midpoint and tangent angle along a ring at its midpoint.
 */
function computeLabelPlacement(ring: [number, number][]): { position: [number, number]; angle: number } {
  if (ring.length < 2) {
    return { position: ring[0] ?? [0, 0], angle: 0 }
  }

  // Walk the ring to find its total length and the midpoint
  let totalLength = 0
  const segLengths: number[] = []
  for (let i = 1; i < ring.length; i++) {
    const dx = ring[i][0] - ring[i - 1][0]
    const dy = ring[i][1] - ring[i - 1][1]
    const len = Math.sqrt(dx * dx + dy * dy)
    segLengths.push(len)
    totalLength += len
  }

  const halfLen = totalLength / 2
  let walked = 0
  for (let i = 0; i < segLengths.length; i++) {
    if (walked + segLengths[i] >= halfLen) {
      const frac = (halfLen - walked) / segLengths[i]
      const px = ring[i][0] + frac * (ring[i + 1][0] - ring[i][0])
      const py = ring[i][1] + frac * (ring[i + 1][1] - ring[i][1])
      const dx = ring[i + 1][0] - ring[i][0]
      const dy = ring[i + 1][1] - ring[i][1]
      const angle = Math.atan2(dy, dx) * (180 / Math.PI)
      return { position: [px, py], angle }
    }
    walked += segLengths[i]
  }

  // Fallback: use midpoint index
  const mid = Math.floor(ring.length / 2)
  const dx = (ring[Math.min(mid + 1, ring.length - 1)][0] - ring[Math.max(mid - 1, 0)][0])
  const dy = (ring[Math.min(mid + 1, ring.length - 1)][1] - ring[Math.max(mid - 1, 0)][1])
  return { position: ring[mid], angle: Math.atan2(dy, dx) * (180 / Math.PI) }
}

/**
 * Generate contour lines from a GridData object.
 * @param grid - The grid data (row-major values array)
 * @param levels - Number of contour levels to generate (default: 10)
 * @returns Array of ContourPath objects with world-space coordinates
 */
export function generateContours(grid: GridData, levels: number = 10): ContourPath[] {
  const { rows, cols, x_min, y_min, dx, dy, values } = grid

  // d3-contour expects column-major layout, but our grid is row-major.
  // d3-contour interprets data as column-major: index = y * width + x
  // Our data is row-major: index = row * cols + col
  // Since d3 uses (x=col, y=row) and our indexing is row*cols+col = y*width+x, it matches.
  const generator = d3Contours().size([cols, rows])

  // Compute threshold values spanning the data range
  let min = Infinity
  let max = -Infinity
  for (const v of values) {
    if (v < min) min = v
    if (v > max) max = v
  }
  if (min === max) return []

  const step = (max - min) / (levels + 1)
  const thresholds: number[] = []
  for (let i = 1; i <= levels; i++) {
    thresholds.push(min + step * i)
  }
  generator.thresholds(thresholds)

  const contourFeatures = generator(values)

  // Convert from grid-space indices to world coordinates
  return contourFeatures.map((feature) => {
    const worldCoords = feature.coordinates.flatMap((polygon) =>
      polygon.map((ring) =>
        ring.map(([gx, gy]): [number, number] => [
          x_min + gx * dx,
          y_min + gy * dy,
        ])
      )
    )

    // Find the longest ring for label placement
    let longestRing = worldCoords[0] ?? []
    let longestLength = 0
    for (const ring of worldCoords) {
      let len = 0
      for (let i = 1; i < ring.length; i++) {
        const rdx = ring[i][0] - ring[i - 1][0]
        const rdy = ring[i][1] - ring[i - 1][1]
        len += Math.sqrt(rdx * rdx + rdy * rdy)
      }
      if (len > longestLength) {
        longestLength = len
        longestRing = ring
      }
    }

    const { position, angle } = computeLabelPlacement(longestRing)

    return {
      value: feature.value,
      coordinates: worldCoords,
      labelPosition: position,
      labelAngle: angle,
    }
  })
}
