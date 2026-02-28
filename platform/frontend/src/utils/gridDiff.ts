/** Grid difference computation for dataset comparison. */

import type { GridData } from '../api'

/** Compute element-wise difference between two grids (A - B).
 *  If grids are aligned (same geometry), subtracts directly.
 *  If mismatched, resamples B onto A's grid using bilinear interpolation.
 */
export function computeDifference(gridA: GridData, gridB: GridData): GridData {
  const aligned =
    gridA.rows === gridB.rows &&
    gridA.cols === gridB.cols &&
    Math.abs(gridA.x_min - gridB.x_min) < 1e-6 &&
    Math.abs(gridA.y_min - gridB.y_min) < 1e-6 &&
    Math.abs(gridA.dx - gridB.dx) < 1e-6 &&
    Math.abs(gridA.dy - gridB.dy) < 1e-6

  if (aligned) {
    const values = gridA.values.map((v, i) => v - gridB.values[i])
    return { ...gridA, values, unit: 'nT (difference)' }
  }

  // Resample B onto A's grid
  const values: number[] = []
  for (let row = 0; row < gridA.rows; row++) {
    for (let col = 0; col < gridA.cols; col++) {
      const x = gridA.x_min + col * gridA.dx
      const y = gridA.y_min + row * gridA.dy
      const valA = gridA.values[row * gridA.cols + col]
      const valB = sampleGrid(gridB, x, y)
      values.push(valA - valB)
    }
  }

  return { ...gridA, values, unit: 'nT (difference)' }
}

function sampleGrid(grid: GridData, x: number, y: number): number {
  const col = (x - grid.x_min) / grid.dx
  const row = (y - grid.y_min) / grid.dy

  const c0 = Math.max(0, Math.min(Math.floor(col), grid.cols - 1))
  const r0 = Math.max(0, Math.min(Math.floor(row), grid.rows - 1))
  const c1 = Math.min(c0 + 1, grid.cols - 1)
  const r1 = Math.min(r0 + 1, grid.rows - 1)

  const fc = Math.max(0, Math.min(col - Math.floor(col), 1))
  const fr = Math.max(0, Math.min(row - Math.floor(row), 1))

  const v00 = grid.values[r0 * grid.cols + c0]
  const v01 = grid.values[r0 * grid.cols + c1]
  const v10 = grid.values[r1 * grid.cols + c0]
  const v11 = grid.values[r1 * grid.cols + c1]

  return v00 * (1 - fc) * (1 - fr) + v01 * fc * (1 - fr) + v10 * (1 - fc) * fr + v11 * fc * fr
}
