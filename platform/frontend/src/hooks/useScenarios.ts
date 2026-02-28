/** TanStack Query hooks for scenario list and detail. */

import { useQuery } from '@tanstack/react-query'
import { fetchScenarios, fetchScenario } from '../api'

export function useScenarios() {
  return useQuery({
    queryKey: ['scenarios'],
    queryFn: fetchScenarios,
  })
}

export function useScenarioDetail(name: string | null) {
  return useQuery({
    queryKey: ['scenario', name],
    queryFn: () => fetchScenario(name!),
    enabled: !!name,
  })
}
