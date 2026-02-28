import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ScenarioSelector } from './ScenarioSelector'
import { TestWrapper } from '../test/wrapper'
import { TEST_SCENARIO_SUMMARY } from '../test/fixtures'
import { useAppStore } from '../stores/appStore'

vi.mock('../hooks/useScenarios', () => ({
  useScenarios: vi.fn(),
  useScenarioDetail: vi.fn(),
}))

import { useScenarios, useScenarioDetail } from '../hooks/useScenarios'
const mockUseScenarios = vi.mocked(useScenarios)
const mockUseScenarioDetail = vi.mocked(useScenarioDetail)

function renderSelector() {
  return render(<ScenarioSelector />, { wrapper: TestWrapper })
}

describe('ScenarioSelector', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useAppStore.setState({ selectedScenario: null })
    mockUseScenarioDetail.mockReturnValue({
      data: undefined,
      isLoading: false,
      error: null,
    } as ReturnType<typeof useScenarioDetail>)
  })

  it('shows loading skeletons while fetching', () => {
    mockUseScenarios.mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
    } as ReturnType<typeof useScenarios>)

    renderSelector()
    const skeletons = document.querySelectorAll('.animate-pulse')
    expect(skeletons.length).toBeGreaterThan(0)
  })

  it('shows error message on failure', () => {
    mockUseScenarios.mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error('Network error'),
    } as ReturnType<typeof useScenarios>)

    renderSelector()
    expect(screen.getByText(/failed to load scenarios/i)).toBeInTheDocument()
  })

  it('shows empty state when no scenarios', () => {
    mockUseScenarios.mockReturnValue({
      data: [],
      isLoading: false,
      error: null,
    } as ReturnType<typeof useScenarios>)

    renderSelector()
    expect(screen.getByText(/no scenarios found/i)).toBeInTheDocument()
  })

  it('renders scenario list', () => {
    const scenarios = [
      TEST_SCENARIO_SUMMARY,
      { ...TEST_SCENARIO_SUMMARY, name: 'Empty Scene', file_name: 'empty', object_count: 0 },
    ]
    mockUseScenarios.mockReturnValue({
      data: scenarios,
      isLoading: false,
      error: null,
    } as ReturnType<typeof useScenarios>)

    renderSelector()
    expect(screen.getByText('Single Ferrous Target')).toBeInTheDocument()
    expect(screen.getByText('Empty Scene')).toBeInTheDocument()
  })

  it('shows object count badge', () => {
    mockUseScenarios.mockReturnValue({
      data: [TEST_SCENARIO_SUMMARY],
      isLoading: false,
      error: null,
    } as ReturnType<typeof useScenarios>)

    renderSelector()
    expect(screen.getByText('1 obj')).toBeInTheDocument()
  })

  it('selects scenario on click', async () => {
    mockUseScenarios.mockReturnValue({
      data: [TEST_SCENARIO_SUMMARY],
      isLoading: false,
      error: null,
    } as ReturnType<typeof useScenarios>)

    renderSelector()
    const user = userEvent.setup()
    await user.click(screen.getByText('Single Ferrous Target'))

    expect(useAppStore.getState().selectedScenario).toBe('single-ferrous-target')
  })
})
