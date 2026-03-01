/** Component gallery — dev-only page for visual testing of all components. */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { TooltipProvider } from '@/components/ui/tooltip'
import { ScenarioSelector } from '../components/ScenarioSelector'
import { ColorScaleControl } from '../components/ColorScaleControl'
import { SimulationParams } from '../components/SimulationParams'
import { StreamControl } from '../components/StreamControl'
import { ImportPanel } from '../components/ImportPanel'
import { Toolbar } from '../components/toolbar/Toolbar'
import { NorthArrow } from '../components/overlays/NorthArrow'
import { MapColorbar } from '../components/overlays/MapColorbar'
import { MapScaleBar } from '../components/overlays/MapScaleBar'
import { PlaybackControl } from '../components/overlays/PlaybackControl'
import { usePlaybackStore } from '../stores/playbackStore'
import { Button } from '@/components/ui/button'

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

      <div className="grid gap-6 max-w-4xl">
        {/* Toolbar (full-width) */}
        <Section title="Toolbar">
          <div className="bg-white border border-zinc-200 rounded-lg overflow-hidden">
            <Toolbar />
          </div>
        </Section>

        {/* shadcn/ui Buttons */}
        <Section title="Buttons (shadcn/ui)">
          <div className="flex flex-wrap gap-2">
            <Button variant="default">Default</Button>
            <Button variant="secondary">Secondary</Button>
            <Button variant="outline">Outline</Button>
            <Button variant="ghost">Ghost</Button>
            <Button variant="destructive">Destructive</Button>
            <Button variant="default" size="sm">Small</Button>
            <Button variant="default" size="lg">Large</Button>
          </div>
        </Section>

        {/* Inner components */}
        <div className="grid gap-6 grid-cols-2">
          <Section title="ScenarioSelector">
            <div className="bg-white border border-zinc-200 rounded-lg">
              <ScenarioSelector />
            </div>
          </Section>

          <Section title="ColorScaleControl">
            <ColorScaleControl />
          </Section>

          <Section title="SimulationParams">
            <SimulationParams />
          </Section>

          <Section title="StreamControl">
            <StreamControl />
          </Section>

          <Section title="ImportPanel">
            <ImportPanel />
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

        {/* Map overlays */}
        <Section title="Map Overlays">
          <div className="relative h-64 bg-zinc-200 rounded-lg overflow-hidden">
            <NorthArrow />
            <MapScaleBar zoom={4} />
            <MapColorbar />
            <div className="absolute inset-0 flex items-center justify-center text-zinc-400 text-sm">
              Map area (overlays positioned absolutely)
            </div>
          </div>
        </Section>

        {/* Playback control */}
        <Section title="PlaybackControl">
          <div className="relative h-20 bg-zinc-200 rounded-lg overflow-hidden">
            <button
              className="absolute top-2 left-2 text-xs text-emerald-600 underline z-20"
              onClick={() => usePlaybackStore.setState({ totalPoints: 200, currentIndex: 42 })}
            >
              Set 200 points (to make control visible)
            </button>
            <PlaybackControl />
          </div>
        </Section>
      </div>
    </div>
  )
}

export default function Gallery() {
  return (
    <QueryClientProvider client={galleryQueryClient}>
      <TooltipProvider>
        <GalleryContent />
      </TooltipProvider>
    </QueryClientProvider>
  )
}
