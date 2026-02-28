/** Profile computation utilities for cross-section views. */

import type { GridData } from '../api'
import type { ProfilePoint } from '../stores/crossSectionStore'

/** Sample grid values along a line using bilinear interpolation. */
export function computeProfile(
  grid: GridData,
  start: [number, number],
  end: [number, number],
  numSamples: number = 200,
): ProfilePoint[] {
  const dx = end[0] - start[0]
  const dy = end[1] - start[1]
  const totalDist = Math.hypot(dx, dy)
  if (totalDist === 0) return []

  const points: ProfilePoint[] = []

  for (let i = 0; i <= numSamples; i++) {
    const frac = i / numSamples
    const x = start[0] + frac * dx
    const y = start[1] + frac * dy
    const distance = frac * totalDist

    const val = sampleGrid(grid, x, y)
    points.push({ distance, x, y, gradient_nt: val })
  }

  return points
}

/** Bilinear interpolation of a grid value at (x, y). */
function sampleGrid(grid: GridData, x: number, y: number): number {
  const col = (x - grid.x_min) / grid.dx
  const row = (y - grid.y_min) / grid.dy

  const c0 = Math.floor(col)
  const r0 = Math.floor(row)
  const c1 = Math.min(c0 + 1, grid.cols - 1)
  const r1 = Math.min(r0 + 1, grid.rows - 1)

  const fc = col - c0
  const fr = row - r0

  // Clamp to grid
  const c0c = Math.max(0, Math.min(c0, grid.cols - 1))
  const r0c = Math.max(0, Math.min(r0, grid.rows - 1))
  const c1c = Math.max(0, Math.min(c1, grid.cols - 1))
  const r1c = Math.max(0, Math.min(r1, grid.rows - 1))

  const v00 = grid.values[r0c * grid.cols + c0c]
  const v01 = grid.values[r0c * grid.cols + c1c]
  const v10 = grid.values[r1c * grid.cols + c0c]
  const v11 = grid.values[r1c * grid.cols + c1c]

  return (
    v00 * (1 - fc) * (1 - fr) +
    v01 * fc * (1 - fr) +
    v10 * (1 - fc) * fr +
    v11 * fc * fr
  )
}

/** Export profile data as CSV string. */
export function exportProfileCSV(data: ProfilePoint[]): string {
  const lines = ['distance_m,x,y,gradient_nt']
  for (const p of data) {
    lines.push(`${p.distance.toFixed(3)},${p.x.toFixed(3)},${p.y.toFixed(3)},${p.gradient_nt.toFixed(4)}`)
  }
  return lines.join('\n')
}
