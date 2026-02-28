import { describe, it, expect, beforeEach } from 'vitest'
import { renderHook } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createElement, type ReactNode } from 'react'
import { useDataset } from './useDataset'
import { useAppStore } from '../stores/appStore'
import { TEST_DATASET } from '../test/fixtures'

let queryClient: QueryClient

function Wrapper({ children }: { children: ReactNode }) {
  return createElement(QueryClientProvider, { client: queryClient }, children)
}

describe('useDataset', () => {
  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    useAppStore.setState({ activeDatasetId: null })
  })

  it('returns null dataset when no active ID', () => {
    const { result } = renderHook(() => useDataset(), { wrapper: Wrapper })
    expect(result.current.dataset).toBeNull()
    expect(result.current.gridData).toBeNull()
    expect(result.current.surveyPoints).toBeNull()
    expect(result.current.metadata).toBeNull()
    expect(result.current.isLoaded).toBe(false)
  })

  it('reads dataset from query cache when active ID is set', () => {
    const id = TEST_DATASET.metadata.id
    queryClient.setQueryData(['dataset', id], TEST_DATASET)
    useAppStore.setState({ activeDatasetId: id })

    const { result } = renderHook(() => useDataset(), { wrapper: Wrapper })
    expect(result.current.dataset).toEqual(TEST_DATASET)
    expect(result.current.isLoaded).toBe(true)
  })

  it('derives gridData from dataset', () => {
    const id = TEST_DATASET.metadata.id
    queryClient.setQueryData(['dataset', id], TEST_DATASET)
    useAppStore.setState({ activeDatasetId: id })

    const { result } = renderHook(() => useDataset(), { wrapper: Wrapper })
    expect(result.current.gridData).toEqual(TEST_DATASET.grid_data)
  })

  it('derives surveyPoints from dataset', () => {
    const id = TEST_DATASET.metadata.id
    queryClient.setQueryData(['dataset', id], TEST_DATASET)
    useAppStore.setState({ activeDatasetId: id })

    const { result } = renderHook(() => useDataset(), { wrapper: Wrapper })
    expect(result.current.surveyPoints).toEqual(TEST_DATASET.survey_points)
  })

  it('derives metadata from dataset', () => {
    const id = TEST_DATASET.metadata.id
    queryClient.setQueryData(['dataset', id], TEST_DATASET)
    useAppStore.setState({ activeDatasetId: id })

    const { result } = renderHook(() => useDataset(), { wrapper: Wrapper })
    expect(result.current.metadata).toEqual(TEST_DATASET.metadata)
  })

  it('returns undefined dataset when ID set but cache is empty', () => {
    useAppStore.setState({ activeDatasetId: 'nonexistent-id' })

    const { result } = renderHook(() => useDataset(), { wrapper: Wrapper })
    // getQueryData returns undefined on cache miss
    expect(result.current.dataset).toBeUndefined()
    // Derived fields still fall back to null via ??
    expect(result.current.gridData).toBeNull()
    expect(result.current.surveyPoints).toBeNull()
    expect(result.current.metadata).toBeNull()
  })

  it('exposes setActiveDatasetId', () => {
    const { result } = renderHook(() => useDataset(), { wrapper: Wrapper })
    expect(typeof result.current.setActiveDatasetId).toBe('function')
  })
})
