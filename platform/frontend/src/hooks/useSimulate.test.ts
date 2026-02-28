import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { useSimulate } from './useSimulate'
import { TestWrapper } from '../test/wrapper'
import { TEST_DATASET } from '../test/fixtures'
import { useAppStore } from '../stores/appStore'

vi.mock('../api', () => ({
  simulateSurvey: vi.fn(),
}))

import { simulateSurvey } from '../api'
const mockSimulate = vi.mocked(simulateSurvey)

describe('useSimulate', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useAppStore.setState({ activeDatasetId: null })
  })

  it('calls simulateSurvey on mutate', async () => {
    mockSimulate.mockResolvedValue(TEST_DATASET)
    const { result } = renderHook(() => useSimulate(), { wrapper: TestWrapper })

    result.current.mutate({ scenario_name: 'test-scenario' })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(mockSimulate).toHaveBeenCalledWith(
      { scenario_name: 'test-scenario' },
      expect.anything(),
    )
  })

  it('updates activeDatasetId in appStore on success', async () => {
    mockSimulate.mockResolvedValue(TEST_DATASET)
    const { result } = renderHook(() => useSimulate(), { wrapper: TestWrapper })

    result.current.mutate({ scenario_name: 'test-scenario' })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(useAppStore.getState().activeDatasetId).toBe(TEST_DATASET.metadata.id)
  })

  it('exposes isPending while loading', async () => {
    let resolve: (v: typeof TEST_DATASET) => void
    mockSimulate.mockReturnValue(new Promise((r) => { resolve = r }))
    const { result } = renderHook(() => useSimulate(), { wrapper: TestWrapper })

    result.current.mutate({ scenario_name: 'test-scenario' })

    await waitFor(() => expect(result.current.isPending).toBe(true))
    resolve!(TEST_DATASET)
    await waitFor(() => expect(result.current.isPending).toBe(false))
  })

  it('exposes error on failure', async () => {
    mockSimulate.mockRejectedValue(new Error('Network error'))
    const { result } = renderHook(() => useSimulate(), { wrapper: TestWrapper })

    result.current.mutate({ scenario_name: 'bad-scenario' })

    await waitFor(() => expect(result.current.isError).toBe(true))
    expect(result.current.error).toBeInstanceOf(Error)
  })
})
