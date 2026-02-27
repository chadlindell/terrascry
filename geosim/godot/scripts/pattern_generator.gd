## Static utility for generating survey line patterns.
##
## Generates waypoint arrays for common survey patterns (parallel lines, zigzag)
## with optional tie lines for drift correction. All coordinates in GeoSim
## convention (X=East, Y=North, Z=Up).
class_name PatternGenerator


## Generate parallel survey lines across a rectangular area.
## Returns array of PackedVector3Array, each representing one survey line.
static func parallel_lines(area: Rect2, spacing: float, angle_rad: float) -> Array:
	if spacing <= 0:
		return []

	var lines: Array = []
	var width := area.size.x
	var height := area.size.y
	var origin := area.position

	# For N-S lines (angle=0): lines run along Y, spaced along X
	# For E-W lines (angle=PI/2): lines run along X, spaced along Y
	# General case: rotate

	var cos_a := cos(angle_rad)
	var sin_a := sin(angle_rad)

	# Determine sweep range perpendicular to line direction
	# The perpendicular direction is (sin_a, -cos_a)
	# We need to cover the full area, so compute the diagonal
	var diagonal := sqrt(width * width + height * height)
	var n_lines := int(ceil(diagonal / spacing)) + 1

	# Center of area
	var cx := origin.x + width / 2.0
	var cy := origin.y + height / 2.0

	# Generate lines centered on the area
	var half_n := n_lines / 2.0
	for i in range(n_lines):
		var offset := (i - half_n) * spacing

		# Line center point, offset perpendicular to line direction
		var pcx := cx + offset * sin_a
		var pcy := cy - offset * cos_a

		# Line endpoints along the line direction (long enough to cross entire area)
		var half_len := diagonal / 2.0
		var p0 := Vector3(pcx - half_len * cos_a, pcy - half_len * sin_a, 0.0)
		var p1 := Vector3(pcx + half_len * cos_a, pcy + half_len * sin_a, 0.0)

		# Clip to area bounds
		var clipped := _clip_line_to_rect(p0, p1, area)
		if clipped.size() == 2:
			var line := PackedVector3Array()
			# Alternate direction for efficiency (boustrophedon)
			if i % 2 == 0:
				line.append(clipped[0])
				line.append(clipped[1])
			else:
				line.append(clipped[1])
				line.append(clipped[0])
			lines.append(line)

	return lines


## Generate zigzag (boustrophedon) survey lines — same as parallel but connected.
## Returns a single array with one connected line.
static func zigzag(area: Rect2, spacing: float, angle_rad: float) -> Array:
	var parallel := parallel_lines(area, spacing, angle_rad)
	if parallel.is_empty():
		return []

	# Connect all lines into one continuous zigzag path
	var zigzag_line := PackedVector3Array()
	for line in parallel:
		for pt in line:
			zigzag_line.append(pt)

	return [zigzag_line]


## Generate perpendicular tie lines for drift correction.
## Tie lines run perpendicular to the main survey lines at regular intervals.
static func add_tie_lines(main_lines: Array, area: Rect2, tie_spacing: float,
		main_angle_rad: float) -> Array:
	if tie_spacing <= 0 or main_lines.is_empty():
		return []

	# Tie lines are perpendicular to main lines
	var tie_angle := main_angle_rad + PI / 2.0
	return parallel_lines(area, tie_spacing, tie_angle)


## Clip a line segment to a rectangle. Returns 0 or 2 points.
static func _clip_line_to_rect(p0: Vector3, p1: Vector3, rect: Rect2) -> Array:
	var x_min := rect.position.x
	var y_min := rect.position.y
	var x_max := rect.position.x + rect.size.x
	var y_max := rect.position.y + rect.size.y

	# Cohen-Sutherland clipping
	var t0 := 0.0
	var t1 := 1.0
	var dx := p1.x - p0.x
	var dy := p1.y - p0.y

	var edges := [
		[-dx, p0.x - x_min],  # Left
		[dx, x_max - p0.x],   # Right
		[-dy, p0.y - y_min],  # Bottom
		[dy, y_max - p0.y],   # Top
	]

	for edge in edges:
		var p: float = edge[0]
		var q: float = edge[1]

		if abs(p) < 1e-10:
			if q < 0:
				return []
		else:
			var r := q / p
			if p < 0:
				t0 = max(t0, r)
			else:
				t1 = min(t1, r)

		if t0 > t1:
			return []

	var clipped_p0 := Vector3(p0.x + t0 * dx, p0.y + t0 * dy, 0.0)
	var clipped_p1 := Vector3(p0.x + t1 * dx, p0.y + t1 * dy, 0.0)

	return [clipped_p0, clipped_p1]
