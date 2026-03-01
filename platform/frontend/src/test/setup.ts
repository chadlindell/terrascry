import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach, vi } from 'vitest'

// Stub framer-motion to avoid animation complexity in tests
vi.mock('framer-motion', () => {
  const React = require('react')
  return {
    motion: new Proxy({}, {
      get: (_: unknown, tag: string) =>
        React.forwardRef((props: Record<string, unknown>, ref: unknown) => {
          // Strip framer-motion-specific props to avoid React DOM warnings
          const filtered = { ...props, ref }
          for (const key of ['layoutId', 'initial', 'animate', 'exit', 'transition', 'variants', 'whileHover', 'whileTap', 'whileInView', 'layout']) {
            delete filtered[key]
          }
          return React.createElement(tag, filtered)
        }),
    }),
    AnimatePresence: ({ children }: { children: React.ReactNode }) => children,
    useAnimation: () => ({ start: () => Promise.resolve(), stop: () => {} }),
    useMotionValue: (v: number) => ({ get: () => v, set: () => {} }),
  }
})

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
