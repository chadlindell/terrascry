/** Color scale state — shared between 2D and 3D views. */

import { create } from 'zustand'
import type { ColormapName } from '../colormap'

interface ColorScaleState {
  colormap: ColormapName
  rangeMin: number
  rangeMax: number
  setColormap: (name: ColormapName) => void
  setRange: (min: number, max: number) => void
}

/** Zustand store for color scale settings shared between 2D and 3D views. */
export const useColorScaleStore = create<ColorScaleState>((set) => ({
  colormap: 'viridis',
  rangeMin: 0,
  rangeMax: 1,
  setColormap: (name) => set({ colormap: name }),
  setRange: (min, max) => set({ rangeMin: min, rangeMax: max }),
}))
