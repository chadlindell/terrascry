import { describe, it, expect } from 'vitest'
import { generateContours, type ContourPath } from './contours'
import { TEST_GRID } from '../test/fixtures'
import type { GridData } from '../api'

/** Helper to build a simple grid with given dimensions and values. */
function makeGrid(
  rows: number,
  cols: number,
  values: number[],
  overrides: Partial<GridData> = {},
): GridData {
  return {
    rows,
    cols,
    x_min: 0,
    y_min: 0,
    dx: 1,
    dy: 1,
    values,
    unit: 'nT',
    ...overrides,
  }
}

describe('generateContours', () => {
  it('returns contour paths for a gradient grid', () => {
    // 5x5 grid with ascending values 0..24
    const values = Array.from({ length: 25 }, (_, i) => i)
    const grid = makeGrid(5, 5, values)
    const result = generateContours(grid, 5)

    expect(result.length).toBeGreaterThan(0)
    result.forEach((cp) => {
      expect(cp).toHaveProperty('value')
      expect(cp).toHaveProperty('coordinates')
      expect(cp.coordinates.length).toBeGreaterThan(0)
    })
  })

  it('respects requested number of levels', () => {
    const values = Array.from({ length: 25 }, (_, i) => i)
    const grid = makeGrid(5, 5, values)

    const result3 = generateContours(grid, 3)
    const result7 = generateContours(grid, 7)

    // Each level produces at most one ContourPath, so count should match levels
    expect(result3.length).toBeLessThanOrEqual(3)
    expect(result7.length).toBeLessThanOrEqual(7)
    // More levels should produce at least as many contours
    expect(result7.length).toBeGreaterThanOrEqual(result3.length)
  })

  it('transforms coordinates to world space', () => {
    const values = Array.from({ length: 25 }, (_, i) => i)
    const grid = makeGrid(5, 5, values, { x_min: 10, y_min: 20, dx: 2, dy: 3 })
    const result = generateContours(grid, 3)

    expect(result.length).toBeGreaterThan(0)
    for (const cp of result) {
      for (const ring of cp.coordinates) {
        for (const [x, y] of ring) {
          // World coords: x_min + gx * dx, y_min + gy * dy
          // d3-contour indices span 0..cols and 0..rows
          expect(x).toBeGreaterThanOrEqual(10)
          expect(x).toBeLessThanOrEqual(10 + 5 * 2) // x_min + cols*dx
          expect(y).toBeGreaterThanOrEqual(20)
          expect(y).toBeLessThanOrEqual(20 + 5 * 3) // y_min + rows*dy
        }
      }
    }
  })

  it('returns empty array for flat grid', () => {
    const values = Array.from({ length: 16 }, () => 5.0)
    const grid = makeGrid(4, 4, values)
    const result = generateContours(grid)

    expect(result).toEqual([])
  })

  it('each ContourPath has value, coordinates, labelPosition, and labelAngle', () => {
    const result = generateContours(TEST_GRID, 3)

    expect(result.length).toBeGreaterThan(0)
    result.forEach((cp: ContourPath) => {
      expect(typeof cp.value).toBe('number')
      expect(Array.isArray(cp.coordinates)).toBe(true)
      cp.coordinates.forEach((ring) => {
        expect(Array.isArray(ring)).toBe(true)
        ring.forEach((coord) => {
          expect(coord).toHaveLength(2)
          expect(typeof coord[0]).toBe('number')
          expect(typeof coord[1]).toBe('number')
        })
      })
      // Label placement fields
      expect(cp.labelPosition).toHaveLength(2)
      expect(typeof cp.labelPosition[0]).toBe('number')
      expect(typeof cp.labelPosition[1]).toBe('number')
      expect(typeof cp.labelAngle).toBe('number')
    })
  })

  it('handles single-level contour', () => {
    const values = Array.from({ length: 25 }, (_, i) => i)
    const grid = makeGrid(5, 5, values)
    const result = generateContours(grid, 1)

    // Single level → single threshold at midpoint
    expect(result).toHaveLength(1)
    // Threshold should be at midpoint: min + (max - min) / 2 = 0 + 24/2 = 12
    expect(result[0].value).toBe(12)
  })

  it('works with negative values', () => {
    const values = Array.from({ length: 25 }, (_, i) => i - 12) // -12 to 12
    const grid = makeGrid(5, 5, values)
    const result = generateContours(grid, 4)

    expect(result.length).toBeGreaterThan(0)
    // Some contour values should be negative, some positive
    const hasNegative = result.some((cp) => cp.value < 0)
    const hasPositive = result.some((cp) => cp.value > 0)
    expect(hasNegative).toBe(true)
    expect(hasPositive).toBe(true)
  })
})
