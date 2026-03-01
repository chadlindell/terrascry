/** Animated survey playback transport bar — scrubber, play/pause, speed controls. */

import { useEffect, useRef } from 'react'
import { usePlaybackStore } from '../../stores/playbackStore'
import { Button } from '@/components/ui/button'
import {
  Play, Pause, SkipBack, SkipForward,
  ChevronsLeft, ChevronsRight,
} from 'lucide-react'

const SPEEDS = [10, 25, 50, 100]

export function PlaybackControl() {
  const isPlaying = usePlaybackStore((s) => s.isPlaying)
  const currentIndex = usePlaybackStore((s) => s.currentIndex)
  const totalPoints = usePlaybackStore((s) => s.totalPoints)
  const speed = usePlaybackStore((s) => s.speed)
  const play = usePlaybackStore((s) => s.play)
  const pause = usePlaybackStore((s) => s.pause)
  const reset = usePlaybackStore((s) => s.reset)
  const setIndex = usePlaybackStore((s) => s.setIndex)
  const setSpeed = usePlaybackStore((s) => s.setSpeed)
  const advance = usePlaybackStore((s) => s.advance)

  // Animation loop
  const rafRef = useRef<number>(0)
  const lastTimeRef = useRef<number>(0)
  const accumulatorRef = useRef<number>(0)

  useEffect(() => {
    if (!isPlaying) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      return
    }

    lastTimeRef.current = performance.now()
    accumulatorRef.current = 0

    const tick = (now: number) => {
      const dt = (now - lastTimeRef.current) / 1000
      lastTimeRef.current = now
      accumulatorRef.current += dt

      const interval = 1 / speed
      while (accumulatorRef.current >= interval) {
        accumulatorRef.current -= interval
        if (!advance()) return
      }

      rafRef.current = requestAnimationFrame(tick)
    }

    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [isPlaying, speed, advance])

  if (totalPoints === 0) return null

  return (
    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 glass-panel-strong shadow-overlay rounded-lg px-3 py-2 flex items-center gap-2">
      {/* Transport controls */}
      <Button variant="ghost" size="icon" className="h-7 w-7" onClick={reset}>
        <SkipBack className="h-3.5 w-3.5" />
      </Button>
      <Button variant="ghost" size="icon" className="h-7 w-7"
        onClick={() => setIndex(Math.max(0, currentIndex - 10))}
      >
        <ChevronsLeft className="h-3.5 w-3.5" />
      </Button>

      <Button
        variant="default"
        size="icon"
        className="h-8 w-8"
        onClick={() => isPlaying ? pause() : play()}
      >
        {isPlaying ? (
          <Pause className="h-4 w-4" />
        ) : (
          <Play className="h-4 w-4" />
        )}
      </Button>

      <Button variant="ghost" size="icon" className="h-7 w-7"
        onClick={() => setIndex(Math.min(totalPoints - 1, currentIndex + 10))}
      >
        <ChevronsRight className="h-3.5 w-3.5" />
      </Button>
      <Button variant="ghost" size="icon" className="h-7 w-7"
        onClick={() => setIndex(totalPoints - 1)}
      >
        <SkipForward className="h-3.5 w-3.5" />
      </Button>

      {/* Scrubber */}
      <input
        type="range"
        min={0}
        max={Math.max(0, totalPoints - 1)}
        value={currentIndex}
        onChange={(e) => setIndex(Number(e.target.value))}
        className="w-32 mx-1"
      />

      {/* Counter */}
      <span className="text-xs font-mono text-zinc-600 min-w-[60px] text-center">
        {currentIndex + 1}/{totalPoints}
      </span>

      {/* Speed buttons */}
      <div className="flex gap-0.5 ml-1">
        {SPEEDS.map((s) => (
          <button
            key={s}
            onClick={() => setSpeed(s)}
            className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${
              speed === s
                ? 'bg-emerald-600 text-white'
                : 'bg-zinc-100 text-zinc-500 hover:bg-zinc-200'
            }`}
          >
            {s >= 100 ? `${s / 10}x` : `${s}`}
          </button>
        ))}
      </div>
    </div>
  )
}
