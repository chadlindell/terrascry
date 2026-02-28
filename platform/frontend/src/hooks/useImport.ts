/** Mutation hook for CSV file upload. */

import { useMutation, useQueryClient } from '@tanstack/react-query'
import { uploadDataset, type Dataset } from '../api'
import { useAppStore } from '../stores/appStore'

export function useImport() {
  const queryClient = useQueryClient()
  const setActiveDatasetId = useAppStore((s) => s.setActiveDatasetId)

  return useMutation({
    mutationFn: uploadDataset,
    onSuccess: (dataset: Dataset) => {
      queryClient.setQueryData(['dataset', dataset.metadata.id], dataset)
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      setActiveDatasetId(dataset.metadata.id)
    },
  })
}
