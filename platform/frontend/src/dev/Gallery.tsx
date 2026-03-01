/** Component gallery — dev-only page for visual testing of sidebar components. */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ScenarioSelector } from '../components/ScenarioSelector'
import { RunSurveyButton } from '../components/RunSurveyButton'
import { ColorScaleControl } from '../components/ColorScaleControl'
import { SimulationParams } from '../components/SimulationParams'
import { StreamControl } from '../components/StreamControl'
import { ImportPanel } from '../components/ImportPanel'
import { useAppStore } from '../stores/appStore'

const galleryQueryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 60_000, retry: 1 },
  },
})

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg shadow-card ring-1 ring-zinc-100 overflow-hidden">
      <div className="bg-zinc-50 px-4 py-2 border-b border-zinc-200/60">
        <h2 className="text-[11px] font-semibold text-zinc-400 uppercase tracking-[0.08em]">{title}</h2>
      </div>
      <div className="p-4 bg-white">{children}</div>
    </div>
  )
}

function GalleryContent() {
  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-900 p-8 font-sans antialiased">
      <h1 className="text-2xl font-bold mb-2">TERRASCRY Component Gallery</h1>
      <p className="text-sm text-zinc-500 mb-8">
        Dev-only page for visual testing. Only available when{' '}
        <code className="bg-zinc-100 px-1.5 py-0.5 rounded-md font-mono text-xs">import.meta.env.DEV</code> is true.
      </p>

      <div className="grid gap-6 max-w-2xl">
        <Section title="ScenarioSelector">
          <div className="w-80 bg-white border border-zinc-200 rounded-lg">
            <ScenarioSelector />
          </div>
        </Section>

        <Section title="RunSurveyButton">
          <div className="w-80 space-y-4">
            <div>
              <p className="text-xs text-zinc-500 mb-2">No scenario selected (hidden):</p>
              <RunSurveyButton />
            </div>
            <div>
              <p className="text-xs text-zinc-500 mb-2">With scenario selected:</p>
              <button
                className="text-xs text-emerald-500 underline mb-2"
                onClick={() => useAppStore.getState().setSelectedScenario('single-ferrous-target')}
              >
                Select a scenario first
              </button>
              <RunSurveyButton />
            </div>
          </div>
        </Section>

        <Section title="ColorScaleControl">
          <div className="w-80">
            <ColorScaleControl />
          </div>
        </Section>

        <Section title="SimulationParams">
          <div className="w-80">
            <SimulationParams />
          </div>
        </Section>

        <Section title="StreamControl">
          <div className="w-80">
            <StreamControl />
          </div>
        </Section>

        <Section title="ImportPanel">
          <div className="w-80">
            <ImportPanel />
          </div>
        </Section>

        <Section title="Glass Panels">
          <div className="flex gap-4">
            <div className="w-40 h-24 glass-panel rounded-lg shadow-card flex items-center justify-center text-xs text-zinc-500">
              glass-panel
            </div>
            <div className="w-40 h-24 glass-panel-strong rounded-lg shadow-overlay flex items-center justify-center text-xs text-zinc-500">
              glass-panel-strong
            </div>
          </div>
        </Section>
      </div>
    </div>
  )
}

export default function Gallery() {
  return (
    <QueryClientProvider client={galleryQueryClient}>
      <GalleryContent />
    </QueryClientProvider>
  )
}
