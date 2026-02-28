/** 2D heatmap view using deck.gl with OrthographicView. */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Deck } from '@deck.gl/core'
import { BitmapLayer, PathLayer, ScatterplotLayer } from '@deck.gl/layers'
import { OrthographicView } from '@deck.gl/core'
import { useQueryClient } from '@tanstack/react-query'
import { useAppStore } from '../stores/appStore'
import { useColorScaleStore } from '../stores/colorScaleStore'
import { useStreamStore } from '../stores/streamStore'
import { useCrossSectionStore } from '../stores/crossSectionStore'
import { gridToImageData, getValueRange, COLORMAPS } from '../colormap'
import { computeProfile } from '../utils/profile'
import { useAnomalies } from '../hooks/useAnomalies'
import type { Dataset, AnomalyCell } from '../api'
import type { StreamPoint } from '../types/streaming'

export function MapView() {
  const activeDatasetId = useAppStore((s) => s.activeDatasetId)
  const showAnomalies = useAppStore((s) => s.showAnomalies)
  const queryClient = useQueryClient()
  const colormap = useColorScaleStore((s) => s.colormap)
  const rangeMin = useColorScaleStore((s) => s.rangeMin)
  const rangeMax = useColorScaleStore((s) => s.rangeMax)
  const setRange = useColorScaleStore((s) => s.setRange)

  // Stream state
  const streamPoints = useStreamStore((s) => s.points)
  const streamEnabled = useStreamStore((s) => s.streamEnabled)
  const streamAnomalies = useStreamStore((s) => s.anomalies)

  // Cross-section state
  const csIsDrawing = useCrossSectionStore((s) => s.isDrawing)
  const csStartPoint = useCrossSectionStore((s) => s.startPoint)
  const csEndPoint = useCrossSectionStore((s) => s.endPoint)
  const csProfileData = useCrossSectionStore((s) => s.profileData)
  const csCursorPosition = useCrossSectionStore((s) => s.cursorPosition)
  const csSetStart = useCrossSectionStore((s) => s.setStartPoint)
  const csSetEnd = useCrossSectionStore((s) => s.setEndPoint)
  const csSetDrawing = useCrossSectionStore((s) => s.setIsDrawing)
  const csSetProfile = useCrossSectionStore((s) => s.setProfileData)

  // Anomaly data from REST endpoint
  const { data: fetchedAnomalies } = useAnomalies(
    showAnomalies ? activeDatasetId : null
  )
  const anomalyCells: AnomalyCell[] = fetchedAnomalies ?? []

  const [containerRef, setContainerRef] = useState<HTMLDivElement | null>(null)
  const [deck, setDeck] = useState<Deck | null>(null)
  const [mousePos, setMousePos] = useState<[number, number] | null>(null)

  // Get dataset from TanStack Query cache
  const dataset = activeDatasetId
    ? queryClient.getQueryData<Dataset>(['dataset', activeDatasetId])
    : null

  // Ref for tooltip callback (avoids stale closure)
  const dataRef = useRef<Dataset | null>(null)
  dataRef.current = dataset

  // Auto-set range when new dataset arrives
  useEffect(() => {
    if (dataset) {
      const [min, max] = getValueRange(dataset.grid_data.values)
      setRange(min, max)
    }
  }, [dataset, setRange])

  // Compute cross-section profile when drawing completes
  useEffect(() => {
    if (dataset && csStartPoint && csEndPoint) {
      const profile = computeProfile(dataset.grid_data, csStartPoint, csEndPoint)
      csSetProfile(profile)
    }
  }, [dataset, csStartPoint, csEndPoint, csSetProfile])

  // Build image data from grid + colormap + range
  const imageData = useMemo(() => {
    if (!dataset) return null
    return gridToImageData(dataset.grid_data, colormap, [rangeMin, rangeMax])
  }, [dataset, colormap, rangeMin, rangeMax])

  // Compute grid bounds
  const bounds = useMemo(() => {
    if (!dataset) return null
    const g = dataset.grid_data
    return [g.x_min, g.y_min, g.x_min + g.cols * g.dx, g.y_min + g.rows * g.dy] as [
      number, number, number, number,
    ]
  }, [dataset])

  // Survey path as array of [x, y] coordinates
  const surveyPath = useMemo(() => {
    if (!dataset || dataset.survey_points.length === 0) return null
    return dataset.survey_points.map((p) => [p.x, p.y] as [number, number])
  }, [dataset])

  // Color function for stream points
  const lut = COLORMAPS[colormap]
  const span = rangeMax - rangeMin || 1

  // Click handler for cross-section drawing
  const handleClick = useCallback(
    (info: { coordinate?: number[] }) => {
      if (!csIsDrawing || !info.coordinate) return
      const pt: [number, number] = [info.coordinate[0], info.coordinate[1]]
      if (!csStartPoint) {
        csSetStart(pt)
      } else {
        csSetEnd(pt)
        csSetDrawing(false)
      }
    },
    [csIsDrawing, csStartPoint, csSetStart, csSetEnd, csSetDrawing],
  )

  // Initialize deck.gl
  const initDeck = useCallback(
    (container: HTMLDivElement | null) => {
      setContainerRef(container)
      if (!container) {
        deck?.finalize()
        setDeck(null)
        return
      }

      const newDeck = new Deck({
        parent: container,
        views: new OrthographicView({ id: 'ortho', flipY: false }),
        initialViewState: {
          target: [10, 10, 0],
          zoom: 4,
        },
        controller: true,
        style: { position: 'absolute', inset: '0' },
        parameters: { clearColor: [0.95, 0.95, 0.96, 1] },
        layers: [],
        onClick: (info: { coordinate?: number[] }) => {
          // Delegate to current handler via ref
          clickRef.current?.(info)
        },
        onHover: (info: { coordinate?: number[] }) => {
          if (info.coordinate) {
            setMousePos([info.coordinate[0], info.coordinate[1]])
          }
        },
        getTooltip: ({ coordinate }: { coordinate?: number[] }) => {
          if (!coordinate || !dataRef.current) return null
          const g = dataRef.current.grid_data
          const col = Math.floor((coordinate[0] - g.x_min) / g.dx)
          const row = Math.floor((coordinate[1] - g.y_min) / g.dy)
          if (row < 0 || row >= g.rows || col < 0 || col >= g.cols) return null
          const val = g.values[row * g.cols + col]
          return {
            text: `X: ${coordinate[0].toFixed(1)}m  Y: ${coordinate[1].toFixed(1)}m\n${val.toFixed(1)} nT`,
            style: {
              backgroundColor: '#ffffff',
              color: '#3f3f46',
              fontSize: '12px',
              padding: '6px 10px',
              borderRadius: '4px',
              border: '1px solid #d4d4d8',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
            },
          }
        },
      })
      setDeck(newDeck)
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  )

  // Keep click handler ref fresh
  const clickRef = useRef(handleClick)
  clickRef.current = handleClick

  // Update layers when data changes
  useEffect(() => {
    if (!deck) return

    const layers = []

    if (imageData && bounds) {
      layers.push(
        new BitmapLayer({
          id: 'heatmap',
          image: imageData,
          bounds,
          pickable: true,
        }),
      )
    }

    if (surveyPath && surveyPath.length > 1) {
      layers.push(
        new PathLayer({
          id: 'survey-track',
          data: [{ path: surveyPath }],
          getPath: (d: { path: [number, number][] }) => d.path,
          getColor: [80, 80, 80, 160],
          getWidth: 0.05,
          widthUnits: 'meters' as const,
          pickable: false,
        }),
      )
    }

    // Live stream points overlay
    if (streamEnabled && streamPoints.length > 0) {
      layers.push(
        new ScatterplotLayer<StreamPoint>({
          id: 'stream-points',
          data: streamPoints,
          getPosition: (d) => [d.x, d.y],
          getFillColor: (d) => {
            const t = Math.max(0, Math.min(255, Math.round(((d.gradient_nt - rangeMin) / span) * 255)))
            return [...lut[t], 220] as [number, number, number, number]
          },
          getRadius: 0.15,
          radiusUnits: 'meters' as const,
          pickable: true,
          updateTriggers: {
            getFillColor: [rangeMin, rangeMax, colormap],
          },
        }),
      )
    }

    // Anomaly overlay (static from REST + live from stream)
    if (showAnomalies) {
      const allAnomalies: { x: number; y: number; strength: number }[] = []

      for (const a of anomalyCells) {
        allAnomalies.push({ x: a.x, y: a.y, strength: Math.abs(a.residual_nt) })
      }
      for (const a of streamAnomalies) {
        allAnomalies.push({ x: a.x, y: a.y, strength: Math.abs(a.anomaly_strength_nt) })
      }

      if (allAnomalies.length > 0) {
        layers.push(
          new ScatterplotLayer({
            id: 'anomalies',
            data: allAnomalies,
            getPosition: (d: { x: number; y: number }) => [d.x, d.y],
            getFillColor: [255, 60, 60, 140],
            getRadius: 0.3,
            radiusUnits: 'meters' as const,
            pickable: true,
          }),
        )
      }
    }

    // Cross-section line
    if (csProfileData.length > 0 && csStartPoint && csEndPoint) {
      layers.push(
        new PathLayer({
          id: 'cross-section-line',
          data: [{ path: [csStartPoint, csEndPoint] }],
          getPath: (d: { path: [number, number][] }) => d.path,
          getColor: [255, 200, 50, 200],
          getWidth: 0.08,
          widthUnits: 'meters' as const,
          pickable: false,
        }),
      )

      // Cursor marker on the cross-section line
      if (csCursorPosition !== null) {
        const totalDist = Math.hypot(
          csEndPoint[0] - csStartPoint[0],
          csEndPoint[1] - csStartPoint[1],
        )
        if (totalDist > 0) {
          const frac = csCursorPosition / totalDist
          const cx = csStartPoint[0] + frac * (csEndPoint[0] - csStartPoint[0])
          const cy = csStartPoint[1] + frac * (csEndPoint[1] - csStartPoint[1])
          layers.push(
            new ScatterplotLayer({
              id: 'cross-section-cursor',
              data: [{ x: cx, y: cy }],
              getPosition: (d: { x: number; y: number }) => [d.x, d.y],
              getFillColor: [255, 200, 50, 255],
              getRadius: 0.2,
              radiusUnits: 'meters' as const,
              pickable: false,
            }),
          )
        }
      }
    }

    // Rubber-band line while drawing cross-section
    if (csIsDrawing && csStartPoint && mousePos) {
      layers.push(
        new PathLayer({
          id: 'cross-section-rubber',
          data: [{ path: [csStartPoint, mousePos] }],
          getPath: (d: { path: [number, number][] }) => d.path,
          getColor: [255, 200, 50, 120],
          getWidth: 0.06,
          widthUnits: 'meters' as const,
          getDashArray: [0.2, 0.1],
          pickable: false,
        }),
      )
    }

    deck.setProps({ layers })

    // Fit viewport to bounds on first dataset
    if (bounds) {
      deck.setProps({
        initialViewState: {
          target: [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2, 0],
          zoom: 4,
        },
      })
    }
  }, [
    deck, imageData, bounds, surveyPath,
    streamEnabled, streamPoints, rangeMin, rangeMax, colormap, lut, span,
    showAnomalies, anomalyCells, streamAnomalies,
    csIsDrawing, csStartPoint, csEndPoint, csProfileData, csCursorPosition, mousePos,
  ])

  // Handle container resize
  useEffect(() => {
    if (!containerRef || !deck) return
    const observer = new ResizeObserver(() => {
      deck.setProps({})
      deck.redraw()
    })
    observer.observe(containerRef)
    return () => observer.disconnect()
  }, [containerRef, deck])

  // Cleanup
  useEffect(() => {
    return () => {
      deck?.finalize()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  if (!dataset && streamPoints.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-zinc-500 text-sm">
          Select a scenario and run a survey to view results.
        </p>
      </div>
    )
  }

  return <div ref={initDeck} className="relative w-full h-full" />
}

export default MapView
