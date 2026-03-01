/** Stream status indicator chip with popover for full StreamControl. */

import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Button } from '@/components/ui/button'
import { useStreamStore } from '../../stores/streamStore'
import { StreamControl } from '../StreamControl'

export function StreamIndicator() {
  const streamEnabled = useStreamStore((s) => s.streamEnabled)
  const status = useStreamStore((s) => s.status)

  const isLive = streamEnabled && status === 'connected'

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="sm" className="gap-1.5 text-xs">
          <span
            className={`h-2 w-2 rounded-full ${
              isLive
                ? 'bg-emerald-500 animate-pulse'
                : streamEnabled
                  ? 'bg-amber-500'
                  : 'bg-zinc-300'
            }`}
          />
          {isLive ? 'Live' : 'Offline'}
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-[320px] p-0">
        <StreamControl />
      </PopoverContent>
    </Popover>
  )
}
