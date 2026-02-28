import { useAppStore } from '../stores/appStore'

export function AppShell() {
  const sidebarOpen = useAppStore((s) => s.sidebarOpen)
  const toggleSidebar = useAppStore((s) => s.toggleSidebar)

  return (
    <div
      className="h-screen overflow-hidden bg-zinc-950 text-zinc-100"
      style={{
        display: 'grid',
        gridTemplateColumns: sidebarOpen ? '320px 1fr' : '0px 1fr',
      }}
    >
      {/* Sidebar */}
      <aside
        className={`flex flex-col bg-zinc-900 border-r border-zinc-700/50 overflow-hidden transition-all ${
          sidebarOpen ? 'w-full' : 'w-0'
        }`}
      >
        {/* Header — pinned */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-700/50">
          <h1 className="text-lg font-semibold tracking-tight text-zinc-100">
            TERRASCRY
          </h1>
          <button
            onClick={toggleSidebar}
            className="p-1 rounded text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 transition-colors"
            aria-label="Close sidebar"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
            </svg>
          </button>
        </div>

        {/* Scrollable content area */}
        <div className="flex-1 min-h-0 overflow-y-auto px-4 py-3">
          <p className="text-sm text-zinc-500">
            Select a scenario to begin.
          </p>
        </div>

        {/* Controls — pinned at bottom */}
        <div className="px-4 py-3 border-t border-zinc-700/50">
          <p className="text-xs text-zinc-500">v0.1.0</p>
        </div>
      </aside>

      {/* Main content */}
      <main className="relative flex items-center justify-center bg-zinc-950 overflow-hidden">
        {!sidebarOpen && (
          <button
            onClick={toggleSidebar}
            className="absolute top-3 left-3 p-1.5 rounded bg-zinc-800 text-zinc-400 hover:text-zinc-100 hover:bg-zinc-700 transition-colors z-10"
            aria-label="Open sidebar"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
            </svg>
          </button>
        )}
        <div className="text-center">
          <p className="text-zinc-500 text-sm">
            Visualization area — select a scenario and run a survey.
          </p>
        </div>
      </main>
    </div>
  )
}
