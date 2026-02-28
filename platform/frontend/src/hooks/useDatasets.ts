/** TanStack Query hooks for dataset listing and deletion. */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { fetchDatasets, deleteDataset } from '../api'

/** Fetch all dataset metadata, newest first. */
export function useDatasets() {
  return useQuery({
    queryKey: ['datasets'],
    queryFn: fetchDatasets,
  })
}

/** Mutation to delete a dataset by ID. Invalidates the datasets list on success. */
export function useDeleteDataset() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: deleteDataset,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
    },
  })
}
