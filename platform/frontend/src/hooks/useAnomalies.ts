/** TanStack Query hook for anomaly detection results. */

import { useQuery } from '@tanstack/react-query'
import { fetchAnomalies } from '../api'

export function useAnomalies(datasetId: string | null, thresholdSigma: number = 3.0) {
  return useQuery({
    queryKey: ['anomalies', datasetId, thresholdSigma],
    queryFn: () => fetchAnomalies(datasetId!, thresholdSigma),
    enabled: !!datasetId,
  })
}
