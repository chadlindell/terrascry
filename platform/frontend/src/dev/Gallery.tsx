/** Component gallery — dev-only page for visual testing of sidebar components. */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ScenarioSelector } from '../components/ScenarioSelector'
import { RunSurveyButton } from '../components/RunSurveyButton'
import { ColorScaleControl } from '../components/ColorScaleControl'
import { useAppStore } from '../stores/appStore'

const galleryQueryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 60_000, retry: 1 },
  },
})

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="border border-zinc-700 rounded-lg overflow-hidden">
      <div className="bg-zinc-800 px-4 py-2 border-b border-zinc-700">
        <h2 className="text-sm font-semibold text-zinc-200">{title}</h2>
      </div>
      <div className="p-4">{children}</div>
    </div>
  )
}

function GalleryContent() {
  return (
    <div className="min-h-screen bg-zinc-900 text-zinc-100 p-8">
      <h1 className="text-2xl font-bold mb-2">TERRASCRY Component Gallery</h1>
      <p className="text-sm text-zinc-400 mb-8">
        Dev-only page for visual testing. Only available when{' '}
        <code className="bg-zinc-800 px-1 rounded">import.meta.env.DEV</code> is true.
      </p>

      <div className="grid gap-6 max-w-2xl">
        <Section title="ScenarioSelector">
          <div className="w-80 bg-zinc-900 border border-zinc-700 rounded">
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
                className="text-xs text-emerald-400 underline mb-2"
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
