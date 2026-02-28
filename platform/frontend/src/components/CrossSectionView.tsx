/** Cross-section profile chart — gradient vs distance along a drawn line. */

import { useCallback } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { useCrossSectionStore } from '../stores/crossSectionStore'
import { exportProfileCSV } from '../utils/profile'

export function CrossSectionView() {
  const profileData = useCrossSectionStore((s) => s.profileData)
  const cursorPosition = useCrossSectionStore((s) => s.cursorPosition)
  const setCursorPosition = useCrossSectionStore((s) => s.setCursorPosition)
  const reset = useCrossSectionStore((s) => s.reset)

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
    <div className="flex flex-col h-full bg-white border-t border-zinc-300/50">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-zinc-300/50">
        <span className="text-xs font-medium text-zinc-500">Cross-Section Profile</span>
        <div className="flex gap-2">
          <button
            onClick={handleExport}
            className="px-2 py-0.5 rounded text-[10px] bg-zinc-100 text-zinc-500 hover:text-zinc-800 transition-colors"
          >
            Export CSV
          </button>
          <button
            onClick={reset}
            className="px-2 py-0.5 rounded text-[10px] bg-zinc-100 text-zinc-500 hover:text-zinc-800 transition-colors"
          >
            Close
          </button>
        </div>
      </div>

      {/* Chart */}
      <div className="flex-1 min-h-0 px-2 py-1">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={profileData}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#d4d4d8" />
            <XAxis
              dataKey="distance"
              tick={{ fill: '#71717a', fontSize: 10 }}
              tickFormatter={(v: number) => `${v.toFixed(1)}m`}
              stroke="#a1a1aa"
            />
            <YAxis
              tick={{ fill: '#71717a', fontSize: 10 }}
              tickFormatter={(v: number) => `${v.toFixed(1)}`}
              stroke="#a1a1aa"
              label={{ value: 'nT', angle: -90, position: 'insideLeft', fill: '#71717a', fontSize: 10 }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#ffffff',
                border: '1px solid #d4d4d8',
                borderRadius: '4px',
                fontSize: '11px',
              }}
              labelFormatter={(v: number) => `${v.toFixed(2)}m`}
              formatter={(v: number) => [`${v.toFixed(2)} nT`, 'Gradient']}
            />
            <Line
              type="monotone"
              dataKey="gradient_nt"
              stroke="#10b981"
              dot={false}
              strokeWidth={1.5}
            />
            {cursorPosition !== null && (
              <ReferenceLine x={cursorPosition} stroke="#f59e0b" strokeDasharray="3 3" />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
