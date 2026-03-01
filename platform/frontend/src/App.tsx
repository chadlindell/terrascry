import { lazy, Suspense } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { AppShell } from './components/AppShell'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
    },
  },
})

const Gallery = import.meta.env.DEV
  ? lazy(() => import('./dev/Gallery'))
  : null

function useDevRoute() {
  if (typeof window === 'undefined') return false
  return window.location.pathname === '/dev/gallery'
}

export default function App() {
  const isGallery = useDevRoute()

  if (isGallery && Gallery) {
    return (
      <Suspense fallback={<div className="p-8 text-zinc-500">Loading gallery...</div>}>
        <Gallery />
      </Suspense>
    )
  }

  return (
    <QueryClientProvider client={queryClient}>
      <AppShell />
    </QueryClientProvider>
  )
}
