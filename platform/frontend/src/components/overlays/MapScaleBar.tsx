/** Scale bar overlay for the 2D map view. Picks a nice round length. */

// The scale bar computes real-world meters from the deck.gl zoom level.
// In an OrthographicView, zoom is log2(pixels per world unit).

const NICE_LENGTHS = [0.5, 1, 2, 5, 10, 20, 50, 100]

function pickNiceLength(metersPerPixel: number, targetWidthPx: number): number {
  const targetMeters = metersPerPixel * targetWidthPx
  // Pick the largest nice length that fits within the target
  let best = NICE_LENGTHS[0]
  for (const n of NICE_LENGTHS) {
    if (n <= targetMeters * 1.2) best = n
  }
  return best
}

interface MapScaleBarProps {
  zoom: number
}

export function MapScaleBar({ zoom }: MapScaleBarProps) {
  // In deck.gl OrthographicView, zoom = log2(pixels per world unit)
  // pixels_per_meter = 2^zoom
  const pixelsPerMeter = Math.pow(2, zoom)
  const metersPerPixel = 1 / pixelsPerMeter

  const TARGET_WIDTH_PX = 120
  const niceLength = pickNiceLength(metersPerPixel, TARGET_WIDTH_PX)
  const barWidthPx = niceLength * pixelsPerMeter

  const label = niceLength >= 1 ? `${niceLength} m` : `${(niceLength * 100).toFixed(0)} cm`

  return (
    <div className="absolute bottom-4 left-4 z-10 glass-panel rounded-lg px-3 py-2 shadow-overlay">
      <div className="flex flex-col items-center gap-1">
        <div
          className="relative h-2"
          style={{ width: Math.max(30, Math.min(200, barWidthPx)) }}
        >
          {/* Bar line */}
          <div className="absolute inset-x-0 top-1/2 h-[2px] bg-zinc-800 -translate-y-1/2"
            style={{ boxShadow: '0 0 0 1px rgba(255,255,255,0.8)' }}
          />
          {/* End caps */}
          <div className="absolute left-0 top-0 w-[2px] h-full bg-zinc-800"
            style={{ boxShadow: '0 0 0 1px rgba(255,255,255,0.8)' }}
          />
          <div className="absolute right-0 top-0 w-[2px] h-full bg-zinc-800"
            style={{ boxShadow: '0 0 0 1px rgba(255,255,255,0.8)' }}
          />
        </div>
        <span className="text-[10px] font-mono text-zinc-700">{label}</span>
      </div>
    </div>
  )
}
