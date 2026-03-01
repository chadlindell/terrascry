import { describe, it, expect, beforeEach } from 'vitest'
import { usePlaybackStore } from './playbackStore'

describe('playbackStore', () => {
  beforeEach(() => {
    usePlaybackStore.setState({
      isPlaying: false,
      currentIndex: 0,
      speed: 25,
      totalPoints: 0,
    })
  })

  it('has correct initial state', () => {
    const state = usePlaybackStore.getState()
    expect(state.isPlaying).toBe(false)
    expect(state.currentIndex).toBe(0)
    expect(state.speed).toBe(25)
    expect(state.totalPoints).toBe(0)
  })

  it('play sets isPlaying to true', () => {
    usePlaybackStore.getState().play()
    expect(usePlaybackStore.getState().isPlaying).toBe(true)
  })

  it('pause sets isPlaying to false', () => {
    usePlaybackStore.getState().play()
    usePlaybackStore.getState().pause()
    expect(usePlaybackStore.getState().isPlaying).toBe(false)
  })

  it('reset stops playback and resets index', () => {
    usePlaybackStore.setState({ isPlaying: true, currentIndex: 50 })
    usePlaybackStore.getState().reset()
    expect(usePlaybackStore.getState().isPlaying).toBe(false)
    expect(usePlaybackStore.getState().currentIndex).toBe(0)
  })

  it('setIndex updates currentIndex', () => {
    usePlaybackStore.getState().setIndex(42)
    expect(usePlaybackStore.getState().currentIndex).toBe(42)
  })

  it('setIndex clamps to 0', () => {
    usePlaybackStore.getState().setIndex(-5)
    expect(usePlaybackStore.getState().currentIndex).toBe(0)
  })

  it('setSpeed updates speed', () => {
    usePlaybackStore.getState().setSpeed(100)
    expect(usePlaybackStore.getState().speed).toBe(100)
  })

  it('advance increments currentIndex', () => {
    usePlaybackStore.setState({ totalPoints: 10, currentIndex: 3 })
    const result = usePlaybackStore.getState().advance()
    expect(result).toBe(true)
    expect(usePlaybackStore.getState().currentIndex).toBe(4)
  })

  it('advance stops at last point and returns false', () => {
    usePlaybackStore.setState({ totalPoints: 10, currentIndex: 9, isPlaying: true })
    const result = usePlaybackStore.getState().advance()
    expect(result).toBe(false)
    expect(usePlaybackStore.getState().isPlaying).toBe(false)
  })

  it('setTotalPoints updates total', () => {
    usePlaybackStore.getState().setTotalPoints(200)
    expect(usePlaybackStore.getState().totalPoints).toBe(200)
  })
})
