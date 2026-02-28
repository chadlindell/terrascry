import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { useScenarios, useScenarioDetail } from './useScenarios'
import { TestWrapper } from '../test/wrapper'
import { TEST_SCENARIO_SUMMARY, TEST_SCENARIO_DETAIL } from '../test/fixtures'

vi.mock('../api', () => ({
  fetchScenarios: vi.fn(),
  fetchScenario: vi.fn(),
}))

import { fetchScenarios, fetchScenario } from '../api'
const mockFetchScenarios = vi.mocked(fetchScenarios)
const mockFetchScenario = vi.mocked(fetchScenario)

describe('useScenarios', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('fetches scenario list', async () => {
    mockFetchScenarios.mockResolvedValue([TEST_SCENARIO_SUMMARY])
    const { result } = renderHook(() => useScenarios(), { wrapper: TestWrapper })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data).toEqual([TEST_SCENARIO_SUMMARY])
    expect(mockFetchScenarios).toHaveBeenCalledOnce()
  })

  it('exposes loading state', () => {
    mockFetchScenarios.mockReturnValue(new Promise(() => {}))
    const { result } = renderHook(() => useScenarios(), { wrapper: TestWrapper })
    expect(result.current.isLoading).toBe(true)
  })

  it('exposes error on failure', async () => {
    mockFetchScenarios.mockRejectedValue(new Error('fetch failed'))
    const { result } = renderHook(() => useScenarios(), { wrapper: TestWrapper })
    await waitFor(() => expect(result.current.isError).toBe(true))
  })
})

describe('useScenarioDetail', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('fetches scenario detail when name is provided', async () => {
    mockFetchScenario.mockResolvedValue(TEST_SCENARIO_DETAIL)
    const { result } = renderHook(
      () => useScenarioDetail('single-ferrous-target'),
      { wrapper: TestWrapper },
    )

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data).toEqual(TEST_SCENARIO_DETAIL)
    expect(mockFetchScenario).toHaveBeenCalledWith('single-ferrous-target')
  })

  it('does not fetch when name is null', () => {
    const { result } = renderHook(
      () => useScenarioDetail(null),
      { wrapper: TestWrapper },
    )
    expect(result.current.fetchStatus).toBe('idle')
    expect(mockFetchScenario).not.toHaveBeenCalled()
  })
})
