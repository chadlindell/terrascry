/** Segmented view mode control for the toolbar using shadcn ToggleGroup. */

import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { useAppStore, type ViewMode } from '../../stores/appStore'

const VIEW_MODES: { value: ViewMode; label: string }[] = [
  { value: '2d', label: '2D' },
  { value: 'split', label: 'Split' },
  { value: '3d', label: '3D' },
  { value: 'comparison', label: 'Compare' },
]

export function ViewModeToggle() {
  const viewMode = useAppStore((s) => s.viewMode)
  const setViewMode = useAppStore((s) => s.setViewMode)

  return (
    <ToggleGroup
      type="single"
      value={viewMode}
      onValueChange={(v) => { if (v) setViewMode(v as ViewMode) }}
      size="sm"
      variant="outline"
      className="border border-border rounded-md"
    >
      {VIEW_MODES.map(({ value, label }) => (
        <ToggleGroupItem
          key={value}
          value={value}
          className="px-3 text-xs font-medium data-[state=on]:bg-primary data-[state=on]:text-primary-foreground"
        >
          {label}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  )
}
