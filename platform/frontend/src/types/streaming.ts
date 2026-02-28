/** Streaming data types for real-time survey visualization. */

/** A single streaming survey point with position, gradient, and timestamp. */
export interface StreamPoint {
  x: number
  y: number
  gradient_nt: number
  timestamp: string
}

/** An anomaly detection event from the streaming pipeline. */
export interface AnomalyEvent {
  x: number
  y: number
  anomaly_strength_nt: number
  anomaly_type: string
  confidence: number
  timestamp: string
}

/** Generic WebSocket message envelope matching backend ConnectionManager format. */
export interface WsMessage<T = unknown> {
  type: string
  payload: T
  timestamp: string
}
