/** Hook to manage active dataset selection and provide data to views. */

import { useQueryClient } from '@tanstack/react-query'
import { useAppStore } from '../stores/appStore'
import type { Dataset } from '../api'

/** Hook providing the active dataset and its derived fields (gridData, surveyPoints, metadata). */
export function useDataset() {
  const activeDatasetId = useAppStore((s) => s.activeDatasetId)
  const setActiveDatasetId = useAppStore((s) => s.setActiveDatasetId)
  const queryClient = useQueryClient()

  const dataset = activeDatasetId
    ? queryClient.getQueryData<Dataset>(['dataset', activeDatasetId])
    : null

  return {
    activeDatasetId,
    dataset,
    gridData: dataset?.grid_data ?? null,
    surveyPoints: dataset?.survey_points ?? null,
    metadata: dataset?.metadata ?? null,
    setActiveDatasetId,
    isLoaded: dataset !== null,
  }
}
