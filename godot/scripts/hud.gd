## Heads-up display for survey readings.
##
## Shows current gradient reading, position, survey statistics,
## and connection status to the physics server.
## Mimics the Pathfinder's on-device display.
extends Control

@onready var gradient_label: Label = $GradientLabel
@onready var position_label: Label = $PositionLabel
@onready var status_label: Label = $StatusLabel

var _reading_history: Array[float] = []
var _max_readings := 500  # Rolling window


func _ready() -> void:
	# Create UI elements if not in scene tree
	if not gradient_label:
		_create_hud()

	# Listen for connection state changes
	PhysicsClient.connection_state_changed.connect(_on_connection_changed)

	# Set initial connection status
	_update_connection_status(PhysicsClient._connected)


func _on_connection_changed(connected: bool) -> void:
	_update_connection_status(connected)


func _update_connection_status(connected: bool) -> void:
	if not status_label:
		return
	if connected:
		status_label.text = "GeoSim LIVE"
		status_label.add_theme_color_override("font_color", Color.GREEN)
	else:
		status_label.text = "GeoSim OFFLINE"
		status_label.add_theme_color_override("font_color", Color.RED)


func update_reading(gradient: float, world_pos: Vector3) -> void:
	_reading_history.append(gradient)
	if _reading_history.size() > _max_readings:
		_reading_history.pop_front()

	if gradient_label:
		# Convert to nT equivalent for display
		var gradient_nT := gradient * 1e9
		gradient_label.text = "Gradient: %.1f nT" % gradient_nT

	if position_label:
		position_label.text = "Pos: (%.1f, %.1f) m" % [world_pos.x, world_pos.z]


func _create_hud() -> void:
	# Gradient reading (large, center-top)
	gradient_label = Label.new()
	gradient_label.name = "GradientLabel"
	gradient_label.text = "Gradient: -- nT"
	gradient_label.position = Vector2(20, 20)
	gradient_label.add_theme_font_size_override("font_size", 24)
	gradient_label.add_theme_color_override("font_color", Color.WHITE)
	add_child(gradient_label)

	# Position (bottom-left)
	position_label = Label.new()
	position_label.name = "PositionLabel"
	position_label.text = "Pos: (--, --) m"
	position_label.position = Vector2(20, 60)
	position_label.add_theme_font_size_override("font_size", 16)
	position_label.add_theme_color_override("font_color", Color.LIGHT_GRAY)
	add_child(position_label)

	# Status (connection indicator)
	status_label = Label.new()
	status_label.name = "StatusLabel"
	status_label.text = "GeoSim OFFLINE"
	status_label.position = Vector2(20, 100)
	status_label.add_theme_font_size_override("font_size", 14)
	status_label.add_theme_color_override("font_color", Color.RED)
	add_child(status_label)
