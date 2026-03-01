import { describe, it, expect, beforeEach } from 'vitest'
import { useCrossSectionStore } from './crossSectionStore'
import type { ProfilePoint } from './crossSectionStore'

describe('crossSectionStore', () => {
  beforeEach(() => {
    useCrossSectionStore.getState().reset()
  })

  it('has correct defaults', () => {
    const state = useCrossSectionStore.getState()
    expect(state.startPoint).toBeNull()
    expect(state.endPoint).toBeNull()
    expect(state.isDrawing).toBe(false)
    expect(state.profileData).toEqual([])
    expect(state.cursorPosition).toBeNull()
  })

  it('setStartPoint updates start point', () => {
    useCrossSectionStore.getState().setStartPoint([1, 2])
    expect(useCrossSectionStore.getState().startPoint).toEqual([1, 2])
  })

  it('setEndPoint updates end point', () => {
    useCrossSectionStore.getState().setEndPoint([3, 4])
    expect(useCrossSectionStore.getState().endPoint).toEqual([3, 4])
  })

  it('setIsDrawing updates drawing state', () => {
    useCrossSectionStore.getState().setIsDrawing(true)
    expect(useCrossSectionStore.getState().isDrawing).toBe(true)
    useCrossSectionStore.getState().setIsDrawing(false)
    expect(useCrossSectionStore.getState().isDrawing).toBe(false)
  })

  it('setProfileData updates profile data', () => {
    const data: ProfilePoint[] = [
      { distance: 0, x: 0, y: 0, gradient_nt: 10 },
      { distance: 1, x: 1, y: 0, gradient_nt: 20 },
    ]
    useCrossSectionStore.getState().setProfileData(data)
    expect(useCrossSectionStore.getState().profileData).toEqual(data)
  })

  it('setCursorPosition updates cursor position', () => {
    useCrossSectionStore.getState().setCursorPosition(0.5)
    expect(useCrossSectionStore.getState().cursorPosition).toBe(0.5)
  })

  it('setCursorPosition can be cleared to null', () => {
    useCrossSectionStore.getState().setCursorPosition(0.5)
    useCrossSectionStore.getState().setCursorPosition(null)
    expect(useCrossSectionStore.getState().cursorPosition).toBeNull()
  })

  it('reset clears all state', () => {
    const store = useCrossSectionStore.getState()
    store.setStartPoint([1, 2])
    store.setEndPoint([3, 4])
    store.setIsDrawing(true)
    store.setProfileData([{ distance: 0, x: 0, y: 0, gradient_nt: 10 }])
    store.setCursorPosition(0.5)

    useCrossSectionStore.getState().reset()

    const state = useCrossSectionStore.getState()
    expect(state.startPoint).toBeNull()
    expect(state.endPoint).toBeNull()
    expect(state.isDrawing).toBe(false)
    expect(state.profileData).toEqual([])
    expect(state.cursorPosition).toBeNull()
  })
})
