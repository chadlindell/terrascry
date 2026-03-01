import * as React from "react"

import { cn } from "@/lib/utils"

interface SliderProps extends React.InputHTMLAttributes<HTMLInputElement> {
  onValueChange?: (value: number) => void
}

const Slider = React.forwardRef<HTMLInputElement, SliderProps>(
  ({ className, onValueChange, onChange, ...props }, ref) => {
    return (
      <input
        type="range"
        className={cn("w-full cursor-pointer", className)}
        ref={ref}
        onChange={(e) => {
          onChange?.(e)
          onValueChange?.(Number(e.target.value))
        }}
        {...props}
      />
    )
  }
)
Slider.displayName = "Slider"

export { Slider }
