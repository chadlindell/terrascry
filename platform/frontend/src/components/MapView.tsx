/** 2D heatmap view using deck.gl with OrthographicView + publication-grade overlays. */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Deck } from '@deck.gl/core'
import { BitmapLayer, PathLayer, ScatterplotLayer, TextLayer } from '@deck.gl/layers'
import { OrthographicView } from '@deck.gl/core'
import { useQueryClient } from '@tanstack/react-query'
import { useAppStore } from '../stores/appStore'
import { useColorScaleStore } from '../stores/colorScaleStore'
import { useStreamStore } from '../stores/streamStore'
import { useCrossSectionStore } from '../stores/crossSectionStore'
import { gridToImageData, getValueRange, COLORMAPS } from '../colormap'
import { computeProfile } from '../utils/profile'
import { useAnomalies } from '../hooks/useAnomalies'
import { generateContours } from '../utils/contours'
import { MapColorbar } from './overlays/MapColorbar'
import { MapScaleBar } from './overlays/MapScaleBar'
import { NorthArrow } from './overlays/NorthArrow'
import type { Dataset, AnomalyCell } from '../api'
import type { StreamPoint } from '../types/streaming'

export function MapView() {
  const activeDatasetId = useAppStore((s) => s.activeDatasetId)
  const showAnomalies = useAppStore((s) => s.showAnomalies)
  const showContours = useAppStore((s) => s.showContours)
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
  const anomalyCells = useMemo<AnomalyCell[]>(
    () => fetchedAnomalies ?? [],
    [fetchedAnomalies],
  )

  const [containerRef, setContainerRef] = useState<HTMLDivElement | null>(null)
  const [deck, setDeck] = useState<Deck | null>(null)
  const [mousePos, setMousePos] = useState<[number, number] | null>(null)
  const [currentZoom, setCurrentZoom] = useState(4)

  // Get dataset from TanStack Query cache
  const dataset = activeDatasetId
    ? queryClient.getQueryData<Dataset>(['dataset', activeDatasetId])
    : null

  // Refs for callbacks (avoids stale closures in deck.gl handlers)
  const dataRef = useRef<Dataset | null>(null)
  dataRef.current = dataset
  const csIsDrawingRef = useRef(csIsDrawing)
  csIsDrawingRef.current = csIsDrawing

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

  // Contour lines from grid data (skip computation when contours are off)
  const contourPaths = useMemo(() => {
    if (!dataset || !showContours) return []
    return generateContours(dataset.grid_data, 12)
  }, [dataset, showContours])

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
        onViewStateChange: ({ viewState }: { viewState: { zoom?: number } }) => {
          if (viewState.zoom !== undefined) {
            setCurrentZoom(viewState.zoom)
          }
          return viewState
        },
        onClick: (info: { coordinate?: number[] }) => {
          clickRef.current?.(info)
        },
        onHover: (info: { coordinate?: number[] }) => {
          if (info.coordinate && csIsDrawingRef.current) {
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
              backgroundColor: 'rgba(255, 255, 255, 0.85)',
              backdropFilter: 'blur(8px)',
              WebkitBackdropFilter: 'blur(8px)',
              color: '#3f3f46',
              fontFamily: "'Geist Mono', monospace",
              fontSize: '11px',
              padding: '6px 10px',
              borderRadius: '8px',
              border: '1px solid rgba(255, 255, 255, 0.5)',
              boxShadow: '0 4px 12px -2px rgba(0,0,0,0.08)',
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

    // Survey path — dashed line for directional cue
    if (surveyPath && surveyPath.length > 1) {
      layers.push(
        new PathLayer({
          id: 'survey-track',
          data: [{ path: surveyPath }],
          getPath: (d: { path: [number, number][] }) => d.path,
          getColor: [60, 60, 60, 180],
          getWidth: 0.04,
          widthUnits: 'meters' as const,
          getDashArray: [0.3, 0.05],
          pickable: false,
        }),
      )
    }

    // Contour overlay with index contours (every 5th thicker)
    if (showContours && contourPaths.length > 0) {
      const contourData = contourPaths.flatMap((cp, cpIdx) =>
        cp.coordinates.map((ring) => ({
          path: ring,
          isIndex: cpIdx % 5 === 0,
        }))
      )
      if (contourData.length > 0) {
        layers.push(
          new PathLayer({
            id: 'contours',
            data: contourData,
            getPath: (d: { path: [number, number][] }) => d.path,
            getColor: (d: { isIndex: boolean }) =>
              d.isIndex ? [30, 30, 30, 120] : [40, 40, 40, 70],
            getWidth: (d: { isIndex: boolean }) =>
              d.isIndex ? 0.04 : 0.015,
            widthUnits: 'meters' as const,
            pickable: false,
            updateTriggers: {
              getColor: [],
              getWidth: [],
            },
          }),
        )
      }

      // Contour labels using TextLayer
      const labelData = contourPaths
        .filter((_, i) => i % 3 === 0) // Label every 3rd contour to avoid clutter
        .filter((cp) => cp.labelPosition[0] !== 0 || cp.labelPosition[1] !== 0)
        .map((cp) => ({
          position: cp.labelPosition,
          text: cp.value.toFixed(1),
          angle: cp.labelAngle,
        }))

      if (labelData.length > 0) {
        layers.push(
          new TextLayer({
            id: 'contour-labels',
            data: labelData,
            getPosition: (d: { position: [number, number] }) => d.position,
            getText: (d: { text: string }) => d.text,
            getAngle: (d: { angle: number }) => -d.angle,
            getSize: 10,
            getColor: [50, 50, 50, 200],
            getTextAnchor: 'middle',
            getAlignmentBaseline: 'center',
            fontFamily: "'Geist Mono', monospace",
            fontWeight: '500',
            background: true,
            getBackgroundColor: [255, 255, 255, 200],
            backgroundPadding: [2, 1],
            pickable: false,
            sizeUnits: 'pixels' as const,
          }),
        )
      }
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

    // Anomaly overlay with strength encoding (size + color)
    if (showAnomalies) {
      const allAnomalies: { x: number; y: number; strength: number; sigma: number }[] = []

      for (const a of anomalyCells) {
        allAnomalies.push({
          x: a.x, y: a.y,
          strength: Math.abs(a.residual_nt),
          sigma: a.sigma,
        })
      }
      for (const a of streamAnomalies) {
        allAnomalies.push({
          x: a.x, y: a.y,
          strength: Math.abs(a.anomaly_strength_nt),
          sigma: 0,
        })
      }

      if (allAnomalies.length > 0) {
        // Compute strength range for normalization
        const maxStrength = Math.max(...allAnomalies.map((a) => a.strength), 1)

        layers.push(
          new ScatterplotLayer({
            id: 'anomalies',
            data: allAnomalies,
            getPosition: (d: { x: number; y: number }) => [d.x, d.y],
            getFillColor: (d: { strength: number }) => {
              const t = Math.min(1, d.strength / maxStrength)
              // Orange (weak) -> deep red (strong)
              const r = Math.round(255)
              const g = Math.round(160 * (1 - t) + 30 * t)
              const b = Math.round(30)
              return [r, g, b, 160] as [number, number, number, number]
            },
            getRadius: (d: { strength: number }) => {
              const t = Math.min(1, d.strength / maxStrength)
              return 0.15 + t * 0.35 // 0.15m to 0.5m
            },
            getLineColor: [180, 40, 40, 200],
            getLineWidth: 0.02,
            stroked: true,
            radiusUnits: 'meters' as const,
            lineWidthUnits: 'meters' as const,
            pickable: true,
            updateTriggers: {
              getFillColor: [allAnomalies.length],
              getRadius: [allAnomalies.length],
            },
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
    showContours, contourPaths,
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
      <div className="flex flex-col items-center justify-center h-full gap-3">
        <div className="w-12 h-12 rounded-full bg-zinc-100 flex items-center justify-center">
          <svg className="w-6 h-6 text-zinc-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 6.75V15m6-6v8.25m.503 3.498l4.875-2.437c.381-.19.622-.58.622-1.006V4.82c0-.836-.88-1.38-1.628-1.006l-3.869 1.934c-.317.159-.69.159-1.006 0L9.503 3.252a1.125 1.125 0 00-1.006 0L3.622 5.689C3.24 5.88 3 6.27 3 6.695V19.18c0 .836.88 1.38 1.628 1.006l3.869-1.934c.317-.159.69-.159 1.006 0l4.994 2.497c.317.158.69.158 1.006 0z" />
          </svg>
        </div>
        <div className="text-center">
          <p className="text-sm font-medium text-zinc-500">No survey data</p>
          <p className="text-xs text-zinc-400 mt-0.5">Select a scenario and run a survey to view results</p>
        </div>
      </div>
    )
  }

  return (
    <div className="relative w-full h-full">
      <div ref={initDeck} className="absolute inset-0" />
      {/* Publication-grade overlays */}
      <NorthArrow />
      <MapScaleBar zoom={currentZoom} />
      <MapColorbar />
    </div>
  )
}

export default MapView
