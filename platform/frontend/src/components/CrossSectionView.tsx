/** Cross-section profile chart — gradient vs distance along a drawn line. */

import { useCallback, useMemo } from 'react'
import { motion } from 'framer-motion'
import {
  LineChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { useCrossSectionStore } from '../stores/crossSectionStore'
import { useColorScaleStore } from '../stores/colorScaleStore'
import { COLORMAPS } from '../colormap'
import { exportProfileCSV } from '../utils/profile'
import { Button } from '@/components/ui/button'
import { Download, X } from 'lucide-react'

/** Build a multi-stop linear gradient from the active colormap LUT. */
function useColormapGradientStops() {
  const colormap = useColorScaleStore((s) => s.colormap)
  return useMemo(() => {
    const lut = COLORMAPS[colormap]
    const stopCount = 8
    const stops: { offset: string; color: string }[] = []
    for (let i = 0; i < stopCount; i++) {
      const t = i / (stopCount - 1)
      const idx = Math.round(t * 255)
      const [r, g, b] = lut[idx]
      stops.push({
        offset: `${(t * 100).toFixed(0)}%`,
        color: `rgb(${r},${g},${b})`,
      })
    }
    return stops
  }, [colormap])
}

export function CrossSectionView() {
  const profileData = useCrossSectionStore((s) => s.profileData)
  const cursorPosition = useCrossSectionStore((s) => s.cursorPosition)
  const setCursorPosition = useCrossSectionStore((s) => s.setCursorPosition)
  const reset = useCrossSectionStore((s) => s.reset)

  const gradientStops = useColormapGradientStops()

  const stats = useMemo(() => {
    if (profileData.length === 0) return null
    const values = profileData.map((p) => p.gradient_nt)
    const min = Math.min(...values)
    const max = Math.max(...values)
    const mean = values.reduce((a, b) => a + b, 0) / values.length
    const range = max - min
    return { min, max, mean, range }
  }, [profileData])

  const handleMouseMove = useCallback(
    (state: { activePayload?: Array<{ payload: { distance: number } }> }) => {
      if (state?.activePayload?.[0]) {
        setCursorPosition(state.activePayload[0].payload.distance)
      }
    },
    [setCursorPosition],
  )

  const handleMouseLeave = useCallback(() => {
    setCursorPosition(null)
  }, [setCursorPosition])

  const handleExport = useCallback(() => {
    const csv = exportProfileCSV(profileData)
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'cross-section-profile.csv'
    a.click()
    URL.revokeObjectURL(url)
  }, [profileData])

  if (profileData.length === 0) return null

  return (
    <motion.div
      initial={{ y: 20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className="flex flex-col h-full bg-white border-t border-zinc-300/50"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-zinc-300/50">
        <div className="flex items-center gap-2">
          <div className="w-0.5 h-4 bg-emerald-500 rounded-full" />
          <span className="text-xs font-medium text-zinc-700">Cross-Section Profile</span>
        </div>
        <div className="flex gap-1.5">
          <Button variant="outline" size="sm" className="h-7 text-xs gap-1" onClick={handleExport}>
            <Download className="h-3 w-3" />
            Export CSV
          </Button>
          <Button variant="ghost" size="sm" className="h-7 text-xs gap-1" onClick={reset}>
            <X className="h-3 w-3" />
            Close
          </Button>
        </div>
      </div>

      {/* Stats row */}
      {stats && (
        <div className="grid grid-cols-4 gap-3 px-3 py-2 border-b border-zinc-100">
          <div>
            <div className="text-[10px] text-zinc-400 uppercase tracking-wider">Min</div>
            <div className="text-sm font-mono text-blue-600">{stats.min.toFixed(1)} nT</div>
          </div>
          <div>
            <div className="text-[10px] text-zinc-400 uppercase tracking-wider">Max</div>
            <div className="text-sm font-mono text-red-600">{stats.max.toFixed(1)} nT</div>
          </div>
          <div>
            <div className="text-[10px] text-zinc-400 uppercase tracking-wider">Mean</div>
            <div className="text-sm font-mono text-zinc-600">{stats.mean.toFixed(1)} nT</div>
          </div>
          <div>
            <div className="text-[10px] text-zinc-400 uppercase tracking-wider">Range</div>
            <div className="text-sm font-mono text-zinc-600">{stats.range.toFixed(1)} nT</div>
          </div>
        </div>
      )}

      {/* Chart */}
      <div className="flex-1 min-h-0 px-2 py-1">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={profileData}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
          >
            <defs>
              <linearGradient id="colormapGradientFill" x1="0" y1="0" x2="0" y2="1">
                {gradientStops.map((stop, i) => (
                  <stop key={i} offset={stop.offset} stopColor={stop.color} stopOpacity={0.3} />
                ))}
              </linearGradient>
              <linearGradient id="colormapStroke" x1="0" y1="1" x2="0" y2="0">
                {gradientStops.map((stop, i) => (
                  <stop key={i} offset={stop.offset} stopColor={stop.color} stopOpacity={1} />
                ))}
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" />
            <XAxis
              dataKey="distance"
              tick={{ fill: '#71717a', fontSize: 11, fontFamily: "'Geist Mono', monospace" }}
              tickFormatter={(v: number) => `${v.toFixed(1)}`}
              stroke="#d4d4d8"
              label={{
                value: 'Distance along profile (m)',
                position: 'insideBottom',
                offset: -5,
                fill: '#71717a',
                fontSize: 11,
                fontFamily: "'Geist Mono', monospace",
              }}
            />
            <YAxis
              width={60}
              tick={{ fill: '#71717a', fontSize: 11, fontFamily: "'Geist Mono', monospace" }}
              tickFormatter={(v: number) => `${v.toFixed(1)}`}
              stroke="#d4d4d8"
              label={{
                value: 'Magnetic Gradient (nT)',
                angle: -90,
                position: 'insideLeft',
                offset: 10,
                fill: '#71717a',
                fontSize: 11,
                fontFamily: "'Geist Mono', monospace",
              }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(255, 255, 255, 0.85)',
                backdropFilter: 'blur(8px)',
                WebkitBackdropFilter: 'blur(8px)',
                border: '1px solid rgba(255, 255, 255, 0.5)',
                borderRadius: '8px',
                fontSize: '11px',
                fontFamily: "'Geist Mono', monospace",
                boxShadow: '0 4px 12px -2px rgba(0,0,0,0.08)',
              }}
              labelFormatter={(v: number) => `${v.toFixed(2)}m`}
              formatter={(v: number) => [`${v.toFixed(2)} nT`, 'Gradient']}
            />
            <Area
              type="monotone"
              dataKey="gradient_nt"
              fill="url(#colormapGradientFill)"
              stroke="none"
            />
            <Line
              type="monotone"
              dataKey="gradient_nt"
              stroke="url(#colormapStroke)"
              dot={false}
              strokeWidth={1.5}
            />
            {cursorPosition !== null && (
              <ReferenceLine x={cursorPosition} stroke="#f59e0b" strokeDasharray="3 3" />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  )
}
