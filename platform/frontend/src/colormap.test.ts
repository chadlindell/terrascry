import { describe, it, expect } from 'vitest'
import { COLORMAPS, getValueRange, gridToImageData } from './colormap'
import type { GridData } from './api'

describe('COLORMAPS', () => {
  it.each(['viridis', 'plasma', 'inferno'] as const)('%s has 256 entries', (name) => {
    expect(COLORMAPS[name]).toHaveLength(256)
  })

  it.each(['viridis', 'plasma', 'inferno'] as const)('%s entries are RGB 0-255', (name) => {
    for (const [r, g, b] of COLORMAPS[name]) {
      expect(r).toBeGreaterThanOrEqual(0)
      expect(r).toBeLessThanOrEqual(255)
      expect(g).toBeGreaterThanOrEqual(0)
      expect(g).toBeLessThanOrEqual(255)
      expect(b).toBeGreaterThanOrEqual(0)
      expect(b).toBeLessThanOrEqual(255)
    }
  })

  it.each(['viridis', 'plasma', 'inferno'] as const)('%s entries are integers', (name) => {
    for (const [r, g, b] of COLORMAPS[name]) {
      expect(Number.isInteger(r)).toBe(true)
      expect(Number.isInteger(g)).toBe(true)
      expect(Number.isInteger(b)).toBe(true)
    }
  })
})

describe('getValueRange', () => {
  it('returns min and max for a normal array', () => {
    expect(getValueRange([3, 1, 4, 1, 5, 9, 2, 6])).toEqual([1, 9])
  })

  it('handles a single value', () => {
    expect(getValueRange([42])).toEqual([42, 42])
  })

  it('handles identical values', () => {
    expect(getValueRange([7, 7, 7])).toEqual([7, 7])
  })

  it('handles negative values', () => {
    expect(getValueRange([-10, -5, -20, -1])).toEqual([-20, -1])
  })

  it('handles mixed positive and negative', () => {
    expect(getValueRange([-3, 0, 5])).toEqual([-3, 5])
  })
})

describe('gridToImageData', () => {
  const grid: GridData = {
    rows: 2,
    cols: 3,
    x_min: 0,
    y_min: 0,
    dx: 1,
    dy: 1,
    values: [0, 50, 100, 100, 50, 0],
    unit: 'nT',
  }

  it('returns ImageData with correct dimensions', () => {
    const img = gridToImageData(grid, 'viridis', [0, 100])
    expect(img.width).toBe(3)
    expect(img.height).toBe(2)
  })

  it('returns RGBA pixels (4 bytes per pixel)', () => {
    const img = gridToImageData(grid, 'viridis', [0, 100])
    expect(img.data.length).toBe(2 * 3 * 4)
  })

  it('sets alpha to 255 for all pixels', () => {
    const img = gridToImageData(grid, 'viridis', [0, 100])
    for (let i = 3; i < img.data.length; i += 4) {
      expect(img.data[i]).toBe(255)
    }
  })

  it('flips Y axis (grid row 0 = bottom → ImageData row 0 = top)', () => {
    // grid row 0 (bottom) = [0, 50, 100], row 1 (top) = [100, 50, 0]
    // ImageData row 0 should be grid row 1 (top)
    const img = gridToImageData(grid, 'viridis', [0, 100])
    // Top-left pixel in image = grid[1][0] = 100 → t=255 → last LUT entry
    const topLeftR = img.data[0]
    const topLeftG = img.data[1]
    const topLeftB = img.data[2]
    expect([topLeftR, topLeftG, topLeftB]).toEqual(COLORMAPS.viridis[255])
    // Bottom-left pixel = grid[0][0] = 0 → t=0 → first LUT entry
    const bottomLeftIdx = 3 * 4 // row 1, col 0
    const bottomLeftR = img.data[bottomLeftIdx]
    const bottomLeftG = img.data[bottomLeftIdx + 1]
    const bottomLeftB = img.data[bottomLeftIdx + 2]
    expect([bottomLeftR, bottomLeftG, bottomLeftB]).toEqual(COLORMAPS.viridis[0])
  })

  it('normalizes values to colormap range', () => {
    const uniformGrid: GridData = {
      rows: 1,
      cols: 2,
      x_min: 0,
      y_min: 0,
      dx: 1,
      dy: 1,
      values: [10, 20],
      unit: 'nT',
    }
    const img = gridToImageData(uniformGrid, 'viridis', [10, 20])
    // val=10 → t=0 → LUT[0], val=20 → t=255 → LUT[255]
    expect([img.data[0], img.data[1], img.data[2]]).toEqual(COLORMAPS.viridis[0])
    expect([img.data[4], img.data[5], img.data[6]]).toEqual(COLORMAPS.viridis[255])
  })

  it('handles zero-span range without NaN', () => {
    const flatGrid: GridData = {
      rows: 1,
      cols: 2,
      x_min: 0,
      y_min: 0,
      dx: 1,
      dy: 1,
      values: [5, 5],
      unit: 'nT',
    }
    const img = gridToImageData(flatGrid, 'plasma', [5, 5])
    // When span=0, code uses span=1: (5-5)/1 = 0 → LUT[0]
    for (let i = 0; i < img.data.length; i += 4) {
      expect(img.data[i + 3]).toBe(255) // alpha valid
    }
    // No NaN in output
    for (let i = 0; i < img.data.length; i++) {
      expect(Number.isNaN(img.data[i])).toBe(false)
    }
  })

  it('works with each colormap', () => {
    const smallGrid: GridData = {
      rows: 1,
      cols: 1,
      x_min: 0,
      y_min: 0,
      dx: 1,
      dy: 1,
      values: [0.5],
      unit: 'nT',
    }
    for (const name of ['viridis', 'plasma', 'inferno'] as const) {
      const img = gridToImageData(smallGrid, name, [0, 1])
      expect(img.width).toBe(1)
      expect(img.height).toBe(1)
      expect(img.data.length).toBe(4)
    }
  })
})
