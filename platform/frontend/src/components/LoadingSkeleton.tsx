/** Loading skeleton — pulsing placeholder for lazy-loaded views. */

export function LoadingSkeleton() {
  return (
    <div className="flex items-center justify-center h-full bg-zinc-800 animate-pulse rounded">
      <p className="text-sm text-zinc-500">Loading...</p>
    </div>
  )
}
