/** Managed WebSocket hook with auto-reconnect and message dispatch to streamStore. */

import { useEffect, useRef, useCallback } from 'react'
import { useStreamStore } from '../stores/streamStore'
import type { WsMessage } from '../types/streaming'
import type { StreamPoint, AnomalyEvent } from '../types/streaming'

interface UseWebSocketOptions {
  enabled?: boolean
}

export function useWebSocket(channel: string, options: UseWebSocketOptions = {}) {
  const { enabled = true } = options
  const wsRef = useRef<WebSocket | null>(null)
  const backoffRef = useRef(1000)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const msgCountRef = useRef(0)
  const rateTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const connectRef = useRef<() => void>(() => {})

  const setStatus = useStreamStore((s) => s.setStatus)
  const addPoint = useStreamStore((s) => s.addPoint)
  const addAnomaly = useStreamStore((s) => s.addAnomaly)
  const updateRate = useStreamStore((s) => s.updateRate)

  const cleanup = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }
    if (pingTimerRef.current) {
      clearInterval(pingTimerRef.current)
      pingTimerRef.current = null
    }
    if (rateTimerRef.current) {
      clearInterval(rateTimerRef.current)
      rateTimerRef.current = null
    }
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
  }, [])

  const connect = useCallback(() => {
    cleanup()

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const url = `${protocol}//${window.location.host}/api/ws/${channel}`

    setStatus('connecting')
    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      setStatus('connected')
      backoffRef.current = 1000 // Reset backoff on success

      // Ping every 30s
      pingTimerRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'ping' }))
        }
      }, 30000)

      // Track message rate (count per second)
      msgCountRef.current = 0
      rateTimerRef.current = setInterval(() => {
        updateRate(msgCountRef.current)
        msgCountRef.current = 0
      }, 1000)
    }

    ws.onmessage = (event) => {
      try {
        const msg: WsMessage = JSON.parse(event.data)
        msgCountRef.current++

        if (msg.type === 'stream_point') {
          addPoint(msg.payload as StreamPoint)
        } else if (msg.type === 'anomaly_detected') {
          addAnomaly(msg.payload as AnomalyEvent)
        }
      } catch {
        // Ignore unparseable messages
      }
    }

    ws.onerror = () => {
      setStatus('error')
    }

    ws.onclose = () => {
      if (pingTimerRef.current) {
        clearInterval(pingTimerRef.current)
        pingTimerRef.current = null
      }
      if (rateTimerRef.current) {
        clearInterval(rateTimerRef.current)
        rateTimerRef.current = null
      }
      wsRef.current = null

      // Auto-reconnect with exponential backoff
      setStatus('disconnected')
      const delay = backoffRef.current
      backoffRef.current = Math.min(backoffRef.current * 2, 30000)
      reconnectTimerRef.current = setTimeout(() => connectRef.current(), delay)
    }
  }, [channel, setStatus, addPoint, addAnomaly, updateRate, cleanup])

  // Keep connectRef in sync so the onclose callback can call the latest version
  useEffect(() => {
    connectRef.current = connect
  }, [connect])

  useEffect(() => {
    if (enabled) {
      connect()
    } else {
      cleanup()
      setStatus('disconnected')
    }
    return cleanup
  }, [enabled, connect, cleanup, setStatus])
}
