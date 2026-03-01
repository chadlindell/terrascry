import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ColorScaleControl } from './ColorScaleControl'
import { useColorScaleStore } from '../stores/colorScaleStore'

describe('ColorScaleControl', () => {
  beforeEach(() => {
    useColorScaleStore.setState({
      colormap: 'viridis',
      rangeMin: 0,
      rangeMax: 100,
    })
  })

  it('renders colormap selector with all options', () => {
    render(<ColorScaleControl />)
    const select = screen.getByRole('combobox')
    expect(select).toHaveValue('viridis')
    expect(screen.getByText('plasma')).toBeInTheDocument()
    expect(screen.getByText('inferno')).toBeInTheDocument()
  })

  it('renders min and max range inputs', () => {
    render(<ColorScaleControl />)
    const inputs = screen.getAllByRole('spinbutton')
    expect(inputs).toHaveLength(2)
  })

  it('updates colormap on selection change', async () => {
    render(<ColorScaleControl />)
    const user = userEvent.setup()
    const select = screen.getByRole('combobox')

    await user.selectOptions(select, 'plasma')
    expect(useColorScaleStore.getState().colormap).toBe('plasma')
  })

  it('updates range min via input', async () => {
    render(<ColorScaleControl />)
    const user = userEvent.setup()
    const inputs = screen.getAllByRole('spinbutton')
    const minInput = inputs[0]

    await user.clear(minInput)
    await user.type(minInput, '25')

    expect(useColorScaleStore.getState().rangeMin).toBe(25)
  })

  it('updates range max via input', async () => {
    render(<ColorScaleControl />)
    const user = userEvent.setup()
    const inputs = screen.getAllByRole('spinbutton')
    const maxInput = inputs[1]

    await user.clear(maxInput)
    await user.type(maxInput, '200')

    expect(useColorScaleStore.getState().rangeMax).toBe(200)
  })

  it('renders gradient bar canvas', () => {
    const { container } = render(<ColorScaleControl />)
    const canvas = container.querySelector('canvas')
    expect(canvas).toBeInTheDocument()
    expect(canvas).toHaveAttribute('width', '200')
    expect(canvas).toHaveAttribute('height', '12')
  })
})
