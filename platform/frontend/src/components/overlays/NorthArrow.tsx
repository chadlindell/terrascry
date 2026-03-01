/** North arrow overlay for the 2D map view. Y+ = North per CLAUDE.md convention. */

export function NorthArrow() {
  return (
    <div className="absolute top-4 left-4 z-10 glass-panel rounded-full w-10 h-10 flex items-center justify-center shadow-overlay">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
        {/* Arrow pointing up (North) */}
        <path
          d="M12 2L8 12h3v10h2V12h3L12 2z"
          fill="#18181b"
          stroke="white"
          strokeWidth="0.5"
        />
        {/* N label */}
        <text
          x="12"
          y="8"
          textAnchor="middle"
          dominantBaseline="central"
          fill="white"
          fontSize="6"
          fontFamily="'Geist Mono', monospace"
          fontWeight="bold"
        >
          N
        </text>
      </svg>
    </div>
  )
}
