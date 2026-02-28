/** TanStack Query hooks for scenario list and detail. */

import { useQuery } from '@tanstack/react-query'
import { fetchScenarios, fetchScenario } from '../api'

/** Query hook for fetching the scenario list. */
export function useScenarios() {
  return useQuery({
    queryKey: ['scenarios'],
    queryFn: fetchScenarios,
  })
}

/** Query hook for fetching full scenario detail. Only fetches when name is non-null. */
export function useScenarioDetail(name: string | null) {
  return useQuery({
    queryKey: ['scenario', name],
    queryFn: () => fetchScenario(name!),
    enabled: !!name,
  })
}
