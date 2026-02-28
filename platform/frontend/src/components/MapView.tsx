/** 2D heatmap view using deck.gl with OrthographicView. */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Deck } from '@deck.gl/core'
import { BitmapLayer, PathLayer } from '@deck.gl/layers'
import { OrthographicView } from '@deck.gl/core'
import { useQueryClient } from '@tanstack/react-query'
import { useAppStore } from '../stores/appStore'
import { useColorScaleStore } from '../stores/colorScaleStore'
import { gridToImageData, getValueRange } from '../colormap'
import type { Dataset } from '../api'

export function MapView() {
  const activeDatasetId = useAppStore((s) => s.activeDatasetId)
  const queryClient = useQueryClient()
  const colormap = useColorScaleStore((s) => s.colormap)
  const rangeMin = useColorScaleStore((s) => s.rangeMin)
  const rangeMax = useColorScaleStore((s) => s.rangeMax)
  const setRange = useColorScaleStore((s) => s.setRange)

  const [containerRef, setContainerRef] = useState<HTMLDivElement | null>(null)
  const [deck, setDeck] = useState<Deck | null>(null)

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
        layers: [],
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
              backgroundColor: '#27272a',
              color: '#e4e4e7',
              fontSize: '12px',
              padding: '6px 10px',
              borderRadius: '4px',
              border: '1px solid #3f3f46',
            },
          }
        },
      })
      setDeck(newDeck)
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  )

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
          getColor: [255, 255, 255, 160],
          getWidth: 0.05,
          widthUnits: 'meters' as const,
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
  }, [deck, imageData, bounds, surveyPath])

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

  if (!dataset) {
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
