import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { RunSurveyButton } from './RunSurveyButton'
import { TestWrapper } from '../test/wrapper'
import { useAppStore } from '../stores/appStore'

const mockMutate = vi.fn()
const mockReset = vi.fn()

vi.mock('../hooks/useSimulate', () => ({
  useSimulate: vi.fn(() => ({
    mutate: mockMutate,
    isPending: false,
    error: null,
    reset: mockReset,
  })),
}))

import { useSimulate } from '../hooks/useSimulate'
const mockUseSimulate = vi.mocked(useSimulate)

function renderButton() {
  return render(<RunSurveyButton />, { wrapper: TestWrapper })
}

describe('RunSurveyButton', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useAppStore.setState({ selectedScenario: null })
    mockUseSimulate.mockReturnValue({
      mutate: mockMutate,
      isPending: false,
      error: null,
      reset: mockReset,
    } as unknown as ReturnType<typeof useSimulate>)
  })

  it('renders nothing when no scenario is selected', () => {
    const { container } = renderButton()
    expect(container.innerHTML).toBe('')
  })

  it('renders button when scenario is selected', () => {
    useAppStore.setState({ selectedScenario: 'test-scenario' })
    renderButton()
    expect(screen.getByRole('button', { name: /run survey/i })).toBeInTheDocument()
  })

  it('shows spinner during pending state', () => {
    useAppStore.setState({ selectedScenario: 'test-scenario' })
    mockUseSimulate.mockReturnValue({
      mutate: mockMutate,
      isPending: true,
      error: null,
      reset: mockReset,
    } as unknown as ReturnType<typeof useSimulate>)

    renderButton()
    expect(screen.getByText(/simulating/i)).toBeInTheDocument()
    expect(screen.getByRole('button')).toBeDisabled()
  })

  it('shows error message on failure', () => {
    useAppStore.setState({ selectedScenario: 'test-scenario' })
    mockUseSimulate.mockReturnValue({
      mutate: mockMutate,
      isPending: false,
      error: new Error('Simulation failed'),
      reset: mockReset,
    } as unknown as ReturnType<typeof useSimulate>)

    renderButton()
    expect(screen.getByText('Simulation failed')).toBeInTheDocument()
  })

  it('calls mutate with scenario name on click', async () => {
    useAppStore.setState({ selectedScenario: 'my-scenario' })
    renderButton()

    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: /run survey/i }))

    expect(mockReset).toHaveBeenCalled()
    expect(mockMutate).toHaveBeenCalledWith({
      scenario_name: 'my-scenario',
      line_spacing: 1,
      sample_spacing: 0.5,
      resolution: 0.5,
    })
  })
})
