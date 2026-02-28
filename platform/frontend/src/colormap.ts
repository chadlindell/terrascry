/** Color mapping utilities for grid data → RGBA ImageData. */

import type { GridData } from './api'

// Viridis colormap LUT (256 entries, RGB 0-255)
const VIRIDIS: [number, number, number][] = []
// Generate viridis approximation using the standard polynomial fit
for (let i = 0; i < 256; i++) {
  const t = i / 255
  // Viridis polynomial approximation (Matplotlib reference)
  const r = Math.round(255 * Math.max(0, Math.min(1,
    0.2777 - 0.0526 * t + 4.0774 * t * t - 15.3583 * t ** 3 + 25.0684 * t ** 4 - 14.8189 * t ** 5)))
  const g = Math.round(255 * Math.max(0, Math.min(1,
    0.0046 + 1.3015 * t - 0.7722 * t * t + 0.3530 * t ** 3 - 0.7532 * t ** 4 + 0.5765 * t ** 5)))
  const b = Math.round(255 * Math.max(0, Math.min(1,
    0.3292 + 1.1524 * t - 5.4069 * t * t + 14.8423 * t ** 3 - 17.4536 * t ** 4 + 7.5780 * t ** 5)))
  VIRIDIS.push([r, g, b])
}

const PLASMA: [number, number, number][] = []
for (let i = 0; i < 256; i++) {
  const t = i / 255
  const r = Math.round(255 * Math.max(0, Math.min(1,
    0.0504 + 2.0146 * t - 1.6154 * t * t - 1.8854 * t ** 3 + 4.6988 * t ** 4 - 2.2548 * t ** 5)))
  const g = Math.round(255 * Math.max(0, Math.min(1,
    0.0298 - 0.1955 * t + 2.2391 * t * t - 5.1647 * t ** 3 + 7.5653 * t ** 4 - 3.5308 * t ** 5)))
  const b = Math.round(255 * Math.max(0, Math.min(1,
    0.5295 + 0.5765 * t - 4.2897 * t * t + 12.4174 * t ** 3 - 16.4380 * t ** 4 + 8.0768 * t ** 5)))
  PLASMA.push([r, g, b])
}

const INFERNO: [number, number, number][] = []
for (let i = 0; i < 256; i++) {
  const t = i / 255
  const r = Math.round(255 * Math.max(0, Math.min(1,
    0.0002 + 0.1592 * t + 4.5321 * t * t - 14.1855 * t ** 3 + 20.1498 * t ** 4 - 9.5986 * t ** 5)))
  const g = Math.round(255 * Math.max(0, Math.min(1,
    0.0003 - 0.0744 * t + 1.6327 * t * t - 4.0666 * t ** 3 + 6.3348 * t ** 4 - 2.8844 * t ** 5)))
  const b = Math.round(255 * Math.max(0, Math.min(1,
    0.0139 + 1.3917 * t - 5.6362 * t * t + 12.6296 * t ** 3 - 14.7629 * t ** 4 + 7.1578 * t ** 5)))
  INFERNO.push([r, g, b])
}

/** Available colormap lookup tables, each with 256 RGB entries. */
export const COLORMAPS = {
  viridis: VIRIDIS,
  plasma: PLASMA,
  inferno: INFERNO,
} as const

/** Union type of available colormap names: 'viridis' | 'plasma' | 'inferno'. */
export type ColormapName = keyof typeof COLORMAPS

/** Compute min and max from grid values. */
export function getValueRange(values: number[]): [number, number] {
  let min = Infinity
  let max = -Infinity
  for (const v of values) {
    if (v < min) min = v
    if (v > max) max = v
  }
  return [min, max]
}

/** Convert GridData values to an RGBA ImageData for BitmapLayer.
 *  Row 0 in grid is y_min (bottom), but ImageData row 0 is top,
 *  so we flip vertically during conversion.
 */
export function gridToImageData(
  grid: GridData,
  colormap: ColormapName,
  range: [number, number],
): ImageData {
  const { rows, cols, values } = grid
  const lut = COLORMAPS[colormap]
  const [min, max] = range
  const span = max - min || 1

  const pixels = new Uint8ClampedArray(rows * cols * 4)

  for (let row = 0; row < rows; row++) {
    // Flip Y: grid row 0 is bottom, ImageData row 0 is top
    const imgRow = rows - 1 - row
    for (let col = 0; col < cols; col++) {
      const val = values[row * cols + col]
      const t = Math.max(0, Math.min(255, Math.round(((val - min) / span) * 255)))
      const [r, g, b] = lut[t]
      const idx = (imgRow * cols + col) * 4
      pixels[idx] = r
      pixels[idx + 1] = g
      pixels[idx + 2] = b
      pixels[idx + 3] = 255
    }
  }

  return new ImageData(pixels, cols, rows)
}
