/** Hook to manage active dataset selection and provide data to views. */

import { useEffect } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useAppStore } from '../stores/appStore'
import { fetchDatasetBinary } from '../api'
import type { Dataset, GridData } from '../api'

/** Attempt to load binary grid data for a dataset, falling back to JSON cache. */
function useBinaryGrid(datasetId: string | null) {
  return useQuery({
    queryKey: ['dataset-binary', datasetId],
    queryFn: () => fetchDatasetBinary(datasetId!),
    enabled: !!datasetId,
    retry: false,
    staleTime: Infinity,
  })
}

/** Hook providing the active dataset and its derived fields (gridData, surveyPoints, metadata). */
export function useDataset() {
  const activeDatasetId = useAppStore((s) => s.activeDatasetId)
  const setActiveDatasetId = useAppStore((s) => s.setActiveDatasetId)
  const queryClient = useQueryClient()

  const dataset = activeDatasetId
    ? queryClient.getQueryData<Dataset>(['dataset', activeDatasetId])
    : null

  // Try binary grid fetch (faster transfer)
  const { data: binaryGrid } = useBinaryGrid(activeDatasetId)

  // If binary grid loaded, update the cached dataset's grid_data
  useEffect(() => {
    if (activeDatasetId && binaryGrid && dataset) {
      const current = queryClient.getQueryData<Dataset>(['dataset', activeDatasetId])
      if (current && current.grid_data !== binaryGrid) {
        queryClient.setQueryData<Dataset>(['dataset', activeDatasetId], {
          ...current,
          grid_data: binaryGrid,
        })
      }
    }
  }, [activeDatasetId, binaryGrid, dataset, queryClient])

  // Use binary grid if available, fall back to JSON grid
  const gridData: GridData | null = binaryGrid ?? dataset?.grid_data ?? null

  return {
    activeDatasetId,
    dataset,
    gridData,
    surveyPoints: dataset?.survey_points ?? null,
    metadata: dataset?.metadata ?? null,
    setActiveDatasetId,
    isLoaded: dataset !== null,
  }
}
