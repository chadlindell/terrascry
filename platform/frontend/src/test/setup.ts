import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

afterEach(() => {
  cleanup()
})

// Stub HTMLCanvasElement.getContext for deck.gl / R3F
HTMLCanvasElement.prototype.getContext = (() => null) as never

// Stub ResizeObserver (not available in jsdom)
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver

// Stub ImageData (not available in jsdom)
if (typeof globalThis.ImageData === 'undefined') {
  globalThis.ImageData = class ImageData {
    readonly data: Uint8ClampedArray
    readonly width: number
    readonly height: number
    constructor(data: Uint8ClampedArray, width: number, height?: number) {
      this.data = data
      this.width = width
      this.height = height ?? data.length / (width * 4)
    }
  } as unknown as typeof ImageData
}
