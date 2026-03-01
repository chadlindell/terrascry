/** Run Survey button with loading spinner and inline error display. */

import { motion, AnimatePresence } from 'framer-motion'
import { useAppStore } from '../stores/appStore'
import { useSimulationParamsStore } from '../stores/simulationParamsStore'
import { useSimulate } from '../hooks/useSimulate'

function Spinner() {
  return (
    <svg
      className="animate-spin h-4 w-4"
      viewBox="0 0 24 24"
      fill="none"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  )
}

export function RunSurveyButton() {
  const selectedScenario = useAppStore((s) => s.selectedScenario)
  const lineSpacing = useSimulationParamsStore((s) => s.lineSpacing)
  const sampleSpacing = useSimulationParamsStore((s) => s.sampleSpacing)
  const resolution = useSimulationParamsStore((s) => s.resolution)
  const { mutate, isPending, error, reset } = useSimulate()

  if (!selectedScenario) return null

  const handleClick = () => {
    reset()
    mutate({
      scenario_name: selectedScenario,
      line_spacing: lineSpacing,
      sample_spacing: sampleSpacing,
      resolution,
    })
  }

  return (
    <div className="space-y-2">
      <button
        onClick={handleClick}
        disabled={isPending}
        className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors bg-emerald-600 hover:bg-emerald-500 text-white disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <AnimatePresence mode="wait">
          {isPending ? (
            <motion.span
              key="loading"
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -5 }}
              transition={{ duration: 0.15 }}
              className="flex items-center gap-2"
            >
              <Spinner />
              Simulating...
            </motion.span>
          ) : (
            <motion.span
              key="idle"
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -5 }}
              transition={{ duration: 0.15 }}
            >
              Run Survey
            </motion.span>
          )}
        </AnimatePresence>
      </button>

      {error && (
        <p className="text-xs text-red-600 px-1">
          {(error as Error).message}
        </p>
      )}
    </div>
  )
}
