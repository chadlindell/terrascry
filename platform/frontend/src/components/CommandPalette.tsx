/** Cmd+K command palette using shadcn CommandDialog (wraps cmdk). */

import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandShortcut,
  CommandSeparator,
} from '@/components/ui/command'
import {
  Map, Play, Upload, Download, Eye, EyeOff,
  Layers, Radio, RadioOff, Trash2, GitCompare,
  Palette, Keyboard, LayoutGrid, Columns2, Box, BarChart3,
} from 'lucide-react'
import { useAppStore, type ViewMode } from '../stores/appStore'
import { useStreamStore } from '../stores/streamStore'
import { useColorScaleStore } from '../stores/colorScaleStore'
import { useCrossSectionStore } from '../stores/crossSectionStore'
import { useScenarios } from '../hooks/useScenarios'
import { useSimulate } from '../hooks/useSimulate'
import { useSimulationParamsStore } from '../stores/simulationParamsStore'
import { useQueryClient } from '@tanstack/react-query'
import type { DatasetMetadata } from '../api'

function useDatasetList() {
  const queryClient = useQueryClient()
  return queryClient.getQueryData<DatasetMetadata[]>(['datasets']) ?? []
}

export function CommandPalette() {
  const open = useAppStore((s) => s.commandPaletteOpen)
  const setOpen = useAppStore((s) => s.setCommandPaletteOpen)
  const setViewMode = useAppStore((s) => s.setViewMode)
  const setSelectedScenario = useAppStore((s) => s.setSelectedScenario)
  const setActiveDatasetId = useAppStore((s) => s.setActiveDatasetId)

  const { data: scenarios } = useScenarios()
  const datasets = useDatasetList()

  const { mutate: simulate } = useSimulate()
  const { lineSpacing, sampleSpacing, resolution } = useSimulationParamsStore()
  const selectedScenario = useAppStore((s) => s.selectedScenario)
  const showContours = useAppStore((s) => s.showContours)
  const showAnomalies = useAppStore((s) => s.showAnomalies)
  const streamEnabled = useStreamStore((s) => s.streamEnabled)

  const run = (fn: () => void) => {
    fn()
    setOpen(false)
  }

  return (
    <CommandDialog open={open} onOpenChange={setOpen}>
      <CommandInput placeholder="Type a command or search..." />
      <CommandList>
        <CommandEmpty>No results found.</CommandEmpty>

        {/* Scenarios */}
        {scenarios && scenarios.length > 0 && (
          <CommandGroup heading="Scenarios">
            {scenarios.map((s) => (
              <CommandItem
                key={s.name}
                value={`scenario ${s.name}`}
                onSelect={() => run(() => setSelectedScenario(s.name))}
              >
                <Map className="mr-2 h-4 w-4" />
                {s.name}
                <span className="ml-2 text-xs text-muted-foreground">
                  {s.object_count} objects
                </span>
              </CommandItem>
            ))}
          </CommandGroup>
        )}

        <CommandSeparator />

        {/* Views */}
        <CommandGroup heading="Views">
          {([
            { mode: '2d' as ViewMode, label: '2D Map', key: '1', icon: LayoutGrid },
            { mode: 'split' as ViewMode, label: 'Split View', key: '2', icon: Columns2 },
            { mode: '3d' as ViewMode, label: '3D Scene', key: '3', icon: Box },
            { mode: 'comparison' as ViewMode, label: 'Comparison', key: '4', icon: BarChart3 },
          ]).map(({ mode, label, key, icon: Icon }) => (
            <CommandItem
              key={mode}
              value={`view ${label}`}
              onSelect={() => run(() => setViewMode(mode))}
            >
              <Icon className="mr-2 h-4 w-4" />
              Switch to {label}
              <CommandShortcut>{key}</CommandShortcut>
            </CommandItem>
          ))}
        </CommandGroup>

        <CommandSeparator />

        {/* Survey */}
        <CommandGroup heading="Survey">
          <CommandItem
            value="run survey"
            disabled={!selectedScenario}
            onSelect={() =>
              run(() => {
                if (!selectedScenario) return
                simulate({
                  scenario_name: selectedScenario,
                  line_spacing: lineSpacing,
                  sample_spacing: sampleSpacing,
                  resolution,
                })
              })
            }
          >
            <Play className="mr-2 h-4 w-4" />
            Run survey
          </CommandItem>
          <CommandItem
            value="import data"
            onSelect={() => {
              setOpen(false)
              // Trigger import dialog through DOM - user can use the toolbar Import button
              document.querySelector<HTMLButtonElement>('[data-import-trigger]')?.click()
            }}
          >
            <Upload className="mr-2 h-4 w-4" />
            Import data
          </CommandItem>
          <CommandItem
            value="export csv"
            onSelect={() => {
              setOpen(false)
              document.querySelector<HTMLButtonElement>('[data-export-trigger]')?.click()
            }}
          >
            <Download className="mr-2 h-4 w-4" />
            Export data
          </CommandItem>
        </CommandGroup>

        <CommandSeparator />

        {/* Overlays */}
        <CommandGroup heading="Overlays">
          <CommandItem
            value="toggle contours"
            onSelect={() => run(() => useAppStore.getState().toggleContours())}
          >
            {showContours ? <EyeOff className="mr-2 h-4 w-4" /> : <Eye className="mr-2 h-4 w-4" />}
            {showContours ? 'Hide' : 'Show'} contours
            <CommandShortcut>C</CommandShortcut>
          </CommandItem>
          <CommandItem
            value="toggle anomalies"
            onSelect={() =>
              run(() => useAppStore.setState((s) => ({ showAnomalies: !s.showAnomalies })))
            }
          >
            {showAnomalies ? <EyeOff className="mr-2 h-4 w-4" /> : <Eye className="mr-2 h-4 w-4" />}
            {showAnomalies ? 'Hide' : 'Show'} anomalies
            <CommandShortcut>A</CommandShortcut>
          </CommandItem>
          <CommandItem
            value="draw cross-section"
            onSelect={() =>
              run(() => {
                const cs = useCrossSectionStore.getState()
                cs.reset()
                cs.setIsDrawing(true)
              })
            }
          >
            <Layers className="mr-2 h-4 w-4" />
            Draw cross-section
            <CommandShortcut>X</CommandShortcut>
          </CommandItem>
        </CommandGroup>

        <CommandSeparator />

        {/* Stream */}
        <CommandGroup heading="Stream">
          <CommandItem
            value="toggle stream"
            onSelect={() => run(() => useStreamStore.getState().toggleStream())}
          >
            {streamEnabled ? <RadioOff className="mr-2 h-4 w-4" /> : <Radio className="mr-2 h-4 w-4" />}
            {streamEnabled ? 'Stop' : 'Start'} live stream
            <CommandShortcut>L</CommandShortcut>
          </CommandItem>
          {streamEnabled && (
            <CommandItem
              value="clear stream data"
              onSelect={() => run(() => useStreamStore.getState().clearPoints())}
            >
              <Trash2 className="mr-2 h-4 w-4" />
              Clear stream data
            </CommandItem>
          )}
        </CommandGroup>

        {/* Datasets */}
        {datasets.length > 0 && (
          <>
            <CommandSeparator />
            <CommandGroup heading="Datasets">
              {datasets.map((d) => (
                <CommandItem
                  key={d.id}
                  value={`dataset ${d.scenario_name} ${d.id}`}
                  onSelect={() => run(() => setActiveDatasetId(d.id))}
                >
                  <GitCompare className="mr-2 h-4 w-4" />
                  {d.scenario_name}
                  <span className="ml-2 text-xs text-muted-foreground">
                    {new Date(d.created_at).toLocaleString()}
                  </span>
                </CommandItem>
              ))}
            </CommandGroup>
          </>
        )}

        <CommandSeparator />

        {/* Display */}
        <CommandGroup heading="Display">
          {(['viridis', 'plasma', 'inferno', 'diverging'] as const).map((cm) => (
            <CommandItem
              key={cm}
              value={`colormap ${cm}`}
              onSelect={() => run(() => useColorScaleStore.getState().setColormap(cm))}
            >
              <Palette className="mr-2 h-4 w-4" />
              Colormap: {cm}
            </CommandItem>
          ))}
        </CommandGroup>

        <CommandSeparator />

        {/* Help */}
        <CommandGroup heading="Help">
          <CommandItem
            value="keyboard shortcuts"
            onSelect={() =>
              run(() => useAppStore.setState({ shortcutLegendOpen: true }))
            }
          >
            <Keyboard className="mr-2 h-4 w-4" />
            Show keyboard shortcuts
            <CommandShortcut>?</CommandShortcut>
          </CommandItem>
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  )
}
