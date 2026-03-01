/** Compact run survey button for the toolbar. */

import { useAppStore } from '../../stores/appStore'
import { useSimulate } from '../../hooks/useSimulate'
import { useSimulationParamsStore } from '../../stores/simulationParamsStore'
import { Button } from '@/components/ui/button'
import { Play, Loader2 } from 'lucide-react'

export function RunSurveyAction() {
  const selectedScenario = useAppStore((s) => s.selectedScenario)
  const { lineSpacing, sampleSpacing, resolution } = useSimulationParamsStore()
  const { mutate, isPending } = useSimulate()

  const handleRun = () => {
    if (!selectedScenario) return
    mutate({
      scenario_name: selectedScenario,
      line_spacing: lineSpacing,
      sample_spacing: sampleSpacing,
      resolution,
    })
  }

  return (
    <Button
      size="sm"
      disabled={!selectedScenario || isPending}
      onClick={handleRun}
      className="gap-1.5"
    >
      {isPending ? (
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
      ) : (
        <Play className="h-3.5 w-3.5" />
      )}
      Run
    </Button>
  )
}
