/** Mutation hook for triggering survey simulation. */

import { useMutation, useQueryClient } from '@tanstack/react-query'
import { simulateSurvey, type Dataset } from '../api'
import { useAppStore } from '../stores/appStore'

/** Mutation hook that triggers a survey simulation, caches the result, and updates the active dataset ID. */
export function useSimulate() {
  const queryClient = useQueryClient()
  const setActiveDatasetId = useAppStore((s) => s.setActiveDatasetId)

  return useMutation({
    mutationFn: simulateSurvey,
    onSuccess: (dataset: Dataset) => {
      // Seed TanStack Query cache with the dataset (avoids storing large arrays in Zustand)
      queryClient.setQueryData(['dataset', dataset.metadata.id], dataset)
      // Store only the ID in Zustand for UI selection
      setActiveDatasetId(dataset.metadata.id)
    },
  })
}
