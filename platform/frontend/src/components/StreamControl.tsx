/** Compact sidebar panel for streaming control — status, start/stop, scenario picker. */

import { useState } from 'react'
import { motion } from 'framer-motion'
import { useStreamStore, type StreamStatus } from '../stores/streamStore'
import { useWebSocket } from '../hooks/useWebSocket'
import { useScenarios } from '../hooks/useScenarios'
import { startStream, stopStream } from '../api'

const STATUS_COLORS: Record<StreamStatus, string> = {
  disconnected: 'bg-zinc-500',
  connecting: 'bg-yellow-500',
  connected: 'bg-emerald-500',
  error: 'bg-red-500',
}

export function StreamControl() {
  const status = useStreamStore((s) => s.status)
  const points = useStreamStore((s) => s.points)
  const anomalies = useStreamStore((s) => s.anomalies)
  const messageRate = useStreamStore((s) => s.messageRate)
  const streamEnabled = useStreamStore((s) => s.streamEnabled)
  const toggleStream = useStreamStore((s) => s.toggleStream)
  const clearPoints = useStreamStore((s) => s.clearPoints)

  const { data: scenarios } = useScenarios()
  const [selectedScenario, setSelectedScenario] = useState('single-ferrous-target')
  const [backendRunning, setBackendRunning] = useState(false)
  const [loading, setLoading] = useState(false)

  // Connect WebSocket to stream channel
  useWebSocket('stream', { enabled: streamEnabled })

  const handleStart = async () => {
    setLoading(true)
    try {
      await startStream(selectedScenario)
      setBackendRunning(true)
    } catch {
      // Error handled silently — status endpoint shows state
    } finally {
      setLoading(false)
    }
  }

  const handleStop = async () => {
    setLoading(true)
    try {
      await stopStream()
      setBackendRunning(false)
    } catch {
      // Ignore
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="px-4 py-3">
      <h2 className="text-[11px] font-semibold text-zinc-400 uppercase tracking-[0.08em] mb-2">
        Live Stream
      </h2>

      {/* Status row */}
      <div className="flex items-center gap-2 mb-2">
        <span className="relative flex h-2 w-2">
          {status === 'connected' && (
            <motion.span
              className="absolute inset-0 rounded-full bg-emerald-400"
              animate={{ scale: [1, 1.8, 1], opacity: [0.6, 0, 0.6] }}
              transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
            />
          )}
          <span className={`relative inline-flex w-2 h-2 rounded-full ${STATUS_COLORS[status]}`} />
        </span>
        <span className="text-xs text-zinc-700 capitalize">{status}</span>
        {status === 'connected' && (
          <span className="text-xs text-zinc-500 ml-auto">{messageRate} msg/s</span>
        )}
      </div>

      {/* Scenario selector (sim mode) */}
      <select
        value={selectedScenario}
        onChange={(e) => setSelectedScenario(e.target.value)}
        disabled={backendRunning}
        className="w-full mb-2 px-2 py-1 rounded-lg bg-white border border-zinc-300 text-xs text-zinc-700 disabled:opacity-50 transition-all duration-150 focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 focus:outline-none"
      >
        {scenarios?.map((s) => (
          <option key={s.file_name} value={s.file_name}>
            {s.name}
          </option>
        )) ?? <option value={selectedScenario}>{selectedScenario}</option>}
      </select>

      {/* Controls */}
      <div className="flex gap-2 mb-2">
        {!backendRunning ? (
          <button
            onClick={handleStart}
            disabled={loading}
            className="flex-1 px-2 py-1 rounded bg-emerald-600 hover:bg-emerald-500 text-xs text-white font-medium disabled:opacity-50 transition-colors"
          >
            {loading ? 'Starting...' : 'Start'}
          </button>
        ) : (
          <button
            onClick={handleStop}
            disabled={loading}
            className="flex-1 px-2 py-1 rounded bg-red-600 hover:bg-red-500 text-xs text-white font-medium disabled:opacity-50 transition-colors"
          >
            {loading ? 'Stopping...' : 'Stop'}
          </button>
        )}
        <button
          onClick={toggleStream}
          className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
            streamEnabled
              ? 'bg-zinc-200 text-zinc-700 hover:bg-zinc-300'
              : 'bg-zinc-100 text-zinc-500 hover:bg-zinc-200'
          }`}
        >
          {streamEnabled ? 'WS On' : 'WS Off'}
        </button>
        <button
          onClick={clearPoints}
          className="px-2 py-1 rounded bg-zinc-100 text-zinc-500 hover:text-zinc-700 text-xs transition-colors"
        >
          Clear
        </button>
      </div>

      {/* Stats */}
      <div className="flex gap-4 text-xs text-zinc-500">
        <span>{points.length} pts</span>
        <span>{anomalies.length} anomalies</span>
      </div>
    </div>
  )
}
