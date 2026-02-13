## Instrument configuration definitions.
##
## Static data class defining properties for each instrument type:
## query command, display channels, colorbar type, units.
## Used by HUD, heatmap overlay, and data recorder.
class_name InstrumentConfig

## Configuration for a single instrument
class Config:
	var query_command: String
	var channels: PackedStringArray
	var heatmap_colorbar: String  # "bipolar", "sequential", "log_sequential"
	var units: String
	var display_name: String
	var sample_rate: float
	var sensor_height: float

	func _init(p_command: String, p_channels: PackedStringArray, p_colorbar: String,
			p_units: String, p_name: String, p_rate: float = 10.0,
			p_height: float = 0.175) -> void:
		query_command = p_command
		channels = p_channels
		heatmap_colorbar = p_colorbar
		units = p_units
		display_name = p_name
		sample_rate = p_rate
		sensor_height = p_height


## Get configuration for an instrument type.
static func get_config(instrument: SurveyManager.Instrument) -> Config:
	match instrument:
		SurveyManager.Instrument.MAG_GRADIOMETER:
			return Config.new(
				"query_gradient",
				PackedStringArray(["gradient_nT"]),
				"bipolar",
				"nT",
				"Mag Gradiometer",
				10.0,
				0.175,
			)
		SurveyManager.Instrument.EM_FDEM:
			return Config.new(
				"query_em_response",
				PackedStringArray(["response_real", "response_imag"]),
				"sequential",
				"ppm",
				"EM (FDEM)",
				5.0,
				0.0,
			)
		SurveyManager.Instrument.RESISTIVITY:
			return Config.new(
				"query_apparent_resistivity",
				PackedStringArray(["apparent_resistivity"]),
				"log_sequential",
				"Ohm-m",
				"Resistivity (ERT)",
				1.0,
				0.0,
			)
		_:
			return Config.new(
				"query_gradient",
				PackedStringArray(["gradient_nT"]),
				"bipolar",
				"nT",
				"Unknown",
			)


## Create a color ramp gradient for the given colorbar type.
static func create_gradient(colorbar_type: String) -> Gradient:
	var grad := Gradient.new()

	match colorbar_type:
		"bipolar":
			# Blue (negative) -> white (zero) -> red (positive)
			grad.colors = PackedColorArray([
				Color(0.1, 0.2, 0.8),
				Color(0.4, 0.5, 0.9),
				Color(0.95, 0.95, 0.95),
				Color(0.9, 0.5, 0.4),
				Color(0.8, 0.2, 0.1),
			])
			grad.offsets = PackedFloat32Array([0.0, 0.25, 0.5, 0.75, 1.0])
		"sequential":
			# Viridis-like: purple -> blue -> green -> yellow
			grad.colors = PackedColorArray([
				Color(0.27, 0.0, 0.33),
				Color(0.13, 0.37, 0.56),
				Color(0.12, 0.63, 0.45),
				Color(0.55, 0.82, 0.22),
				Color(0.99, 0.91, 0.14),
			])
			grad.offsets = PackedFloat32Array([0.0, 0.25, 0.5, 0.75, 1.0])
		"log_sequential":
			# Warm tones for resistivity: brown -> orange -> yellow -> white
			grad.colors = PackedColorArray([
				Color(0.3, 0.15, 0.05),
				Color(0.6, 0.3, 0.1),
				Color(0.9, 0.6, 0.2),
				Color(0.95, 0.85, 0.5),
				Color(1.0, 1.0, 0.9),
			])
			grad.offsets = PackedFloat32Array([0.0, 0.25, 0.5, 0.75, 1.0])
		_:
			# Default rainbow
			grad.colors = PackedColorArray([
				Color(0.1, 0.2, 0.8),
				Color(0.2, 0.7, 0.3),
				Color(0.9, 0.8, 0.2),
				Color(0.9, 0.2, 0.1),
			])
			grad.offsets = PackedFloat32Array([0.0, 0.33, 0.66, 1.0])

	return grad
