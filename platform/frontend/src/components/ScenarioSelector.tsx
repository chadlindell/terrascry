/** Sidebar panel: scenario list + detail section below. */

import { AnimatePresence, motion } from 'framer-motion'
import { useAppStore } from '../stores/appStore'
import { useScenarios, useScenarioDetail } from '../hooks/useScenarios'
import type { ScenarioSummary } from '../api'

function SkeletonRow() {
  return (
    <div className="animate-pulse px-3 py-2.5 space-y-1.5">
      <div className="h-4 bg-zinc-200 rounded w-3/4" />
      <div className="h-3 bg-zinc-100 rounded w-full" />
    </div>
  )
}

function ScenarioRow({
  scenario,
  selected,
  onClick,
}: {
  scenario: ScenarioSummary
  selected: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-3 py-2.5 rounded-lg transition-colors relative ${
        selected
          ? 'bg-white shadow-card ring-1 ring-zinc-100'
          : 'hover:bg-white/60 border-l-2 border-transparent'
      }`}
    >
      {selected && (
        <motion.div
          layoutId="scenario-indicator"
          className="absolute left-0 top-2 bottom-2 w-0.5 bg-emerald-500 rounded-full"
          transition={{ type: 'spring', stiffness: 400, damping: 30 }}
        />
      )}
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-medium text-zinc-900 truncate">
          {scenario.name}
        </span>
        <span className="shrink-0 text-xs text-zinc-500 bg-zinc-200 px-1.5 py-0.5 rounded">
          {scenario.object_count} obj
        </span>
      </div>
      {scenario.description && (
        <p className="text-xs text-zinc-500 mt-0.5 line-clamp-2">
          {scenario.description}
        </p>
      )}
    </button>
  )
}

function ScenarioDetailPanel({ name }: { name: string }) {
  const { data: detail, isLoading, error } = useScenarioDetail(name)

  if (isLoading) {
    return (
      <div className="animate-pulse space-y-2 px-3 py-3">
        <div className="h-4 bg-zinc-200 rounded w-1/2" />
        <div className="h-3 bg-zinc-100 rounded w-full" />
        <div className="h-3 bg-zinc-100 rounded w-3/4" />
        <div className="h-20 bg-zinc-100 rounded" />
      </div>
    )
  }

  if (error) {
    return (
      <p className="px-3 py-3 text-xs text-red-600">
        Failed to load detail: {(error as Error).message}
      </p>
    )
  }

  if (!detail) return null

  return (
    <motion.div
      initial={{ height: 0, opacity: 0 }}
      animate={{ height: 'auto', opacity: 1 }}
      exit={{ height: 0, opacity: 0 }}
      transition={{ duration: 0.25, ease: 'easeInOut' }}
      className="overflow-hidden"
    >
      <div className="px-3 py-3 space-y-3">
        <h3 className="text-sm font-semibold text-zinc-900">{detail.name}</h3>
        {detail.description && (
          <p className="text-xs text-zinc-500">{detail.description}</p>
        )}

        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="bg-white rounded-lg p-2.5 shadow-panel ring-1 ring-zinc-100">
            <span className="text-zinc-400 block">Terrain</span>
            <span className="text-zinc-700">
              {detail.terrain.x_extent[1] - detail.terrain.x_extent[0]}m &times;{' '}
              {detail.terrain.y_extent[1] - detail.terrain.y_extent[0]}m
            </span>
          </div>
          <div className="bg-white rounded-lg p-2.5 shadow-panel ring-1 ring-zinc-100">
            <span className="text-zinc-400 block">Objects</span>
            <span className="text-zinc-700">{detail.objects.length}</span>
          </div>
        </div>

        {detail.objects.length > 0 && (
          <div className="space-y-1">
            <span className="text-xs text-zinc-400 font-medium">Buried objects</span>
            {detail.objects.map((obj) => (
              <div
                key={obj.name}
                className="flex items-center justify-between text-xs bg-white rounded-lg px-2 py-1.5 shadow-panel ring-1 ring-zinc-100"
              >
                <span className="text-zinc-700">{obj.name}</span>
                <span className="text-zinc-400">{obj.object_type}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  )
}

export function ScenarioSelector() {
  const selectedScenario = useAppStore((s) => s.selectedScenario)
  const setSelectedScenario = useAppStore((s) => s.setSelectedScenario)
  const { data: scenarios, isLoading, error } = useScenarios()

  return (
    <div className="flex flex-col min-h-0">
      {/* Section header */}
      <div className="px-3 py-2 border-b border-zinc-300/50">
        <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-[0.08em]">
          Scenarios
        </span>
      </div>

      {/* List */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        {isLoading && (
          <div className="space-y-1 py-1">
            {Array.from({ length: 5 }, (_, i) => (
              <SkeletonRow key={i} />
            ))}
          </div>
        )}

        {error && (
          <p className="px-3 py-3 text-xs text-red-600">
            Failed to load scenarios: {(error as Error).message}
          </p>
        )}

        {scenarios && scenarios.length === 0 && (
          <p className="px-3 py-3 text-xs text-zinc-400">
            No scenarios found.
          </p>
        )}

        {scenarios && (
          <div className="space-y-0.5 py-1">
            {scenarios.map((s) => (
              <ScenarioRow
                key={s.file_name}
                scenario={s}
                selected={selectedScenario === s.file_name}
                onClick={() => setSelectedScenario(s.file_name)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Detail section */}
      <AnimatePresence>
        {selectedScenario && (
          <div className="border-t border-zinc-300/50 overflow-y-auto max-h-[40%]">
            <ScenarioDetailPanel name={selectedScenario} />
          </div>
        )}
      </AnimatePresence>
    </div>
  )
}
