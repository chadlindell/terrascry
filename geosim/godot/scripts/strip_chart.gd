## Strip chart widget for displaying scrolling gradient readings.
##
## Draws a scrolling line graph of recent gradient values with
## auto-ranging, threshold lines, and color-coded magnitude.
extends Control

## Maximum number of data points to display
@export var max_points := 200

## Detection threshold in nT (shown as horizontal lines)
var detection_threshold := 1.0

## Colors
var bg_color := Color(0.1, 0.1, 0.1, 0.8)
var grid_color := Color(0.3, 0.3, 0.3, 0.4)
var threshold_color := Color(1.0, 1.0, 0.0, 0.3)

## Data storage
var _values: PackedFloat64Array = []
var _y_min := -5.0
var _y_max := 5.0


## Add a new gradient reading (in nT) to the chart.
func add_value(value_nT: float) -> void:
	_values.append(value_nT)
	if _values.size() > max_points:
		_values = _values.slice(_values.size() - max_points)

	# Auto-range to fit data with some headroom
	if _values.size() > 0:
		var abs_max := 0.0
		for v in _values:
			abs_max = max(abs_max, abs(v))
		abs_max = max(abs_max, 2.0)  # Minimum range ±2 nT
		_y_min = -abs_max * 1.2
		_y_max = abs_max * 1.2

	queue_redraw()


func _draw() -> void:
	var w := size.x
	var h := size.y

	# Background
	draw_rect(Rect2(Vector2.ZERO, size), bg_color)

	# Zero line
	var zero_y := _val_to_y(0.0, h)
	draw_line(Vector2(0, zero_y), Vector2(w, zero_y), grid_color, 1.0)

	# Threshold lines
	if detection_threshold > 0:
		var thresh_y_pos := _val_to_y(detection_threshold, h)
		var thresh_y_neg := _val_to_y(-detection_threshold, h)
		draw_line(Vector2(0, thresh_y_pos), Vector2(w, thresh_y_pos), threshold_color, 1.0)
		draw_line(Vector2(0, thresh_y_neg), Vector2(w, thresh_y_neg), threshold_color, 1.0)

	# Data line
	if _values.size() < 2:
		return

	var n := _values.size()
	var x_step := w / float(max_points - 1)
	var x_offset := (max_points - n) * x_step

	for i in range(n - 1):
		var x0 := x_offset + i * x_step
		var y0 := _val_to_y(_values[i], h)
		var x1 := x_offset + (i + 1) * x_step
		var y1 := _val_to_y(_values[i + 1], h)
		var color := _gradient_color(_values[i])
		draw_line(Vector2(x0, y0), Vector2(x1, y1), color, 1.5, true)

	# Y-axis range labels
	var font := ThemeDB.fallback_font
	var font_size := 10
	var label_color := Color(0.6, 0.6, 0.6, 0.8)
	draw_string(font, Vector2(4, 12), "%.1f" % _y_max, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, label_color)
	draw_string(font, Vector2(4, h - 4), "%.1f" % _y_min, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, label_color)


func _val_to_y(val: float, height: float) -> float:
	if _y_max <= _y_min:
		return height / 2.0
	var normalized := (val - _y_min) / (_y_max - _y_min)
	return height * (1.0 - normalized)


func _gradient_color(value_nT: float) -> Color:
	var abs_val: float = absf(value_nT)
	if abs_val < 1.0:
		return Color.GREEN
	elif abs_val < 10.0:
		return Color.YELLOW
	elif abs_val < 100.0:
		return Color.ORANGE_RED
	else:
		return Color.RED
