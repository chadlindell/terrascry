/** Loading skeleton — layout-aware pulsing placeholder for lazy-loaded views. */

export function LoadingSkeleton() {
  return (
    <div className="flex flex-col items-center justify-center h-full bg-zinc-50 animate-pulse gap-3">
      <div className="w-12 h-12 rounded-full bg-zinc-200/60" />
      <div className="w-32 h-3 rounded bg-zinc-200/60" />
    </div>
  )
}
