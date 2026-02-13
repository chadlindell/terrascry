## Coordinate transformation utility (autoload singleton).
##
## Single source of truth for all GeoSim <-> Godot coordinate conversions.
## GeoSim uses right-handed: X=East, Y=North, Z=Up
## Godot uses: X=right, Y=up, Z=back (into screen)
##
## Usage from any node:
##   var geosim_pos = CoordUtil.to_geosim(godot_position)
##   var godot_pos = CoordUtil.to_godot(geosim_position)
extends Node

## Mock UTM offset for industry-format display
const UTM_EASTING_OFFSET := 500000.0
const UTM_NORTHING_OFFSET := 4500000.0


## Convert Godot world position to GeoSim coordinates.
## Godot [X, Y_up, Z] -> GeoSim [X_east, Y_north, Z_up]
func to_geosim(godot_pos: Vector3) -> Vector3:
	return Vector3(godot_pos.x, godot_pos.z, godot_pos.y)


## Convert GeoSim coordinates to Godot world position.
## GeoSim [X_east, Y_north, Z_up] -> Godot [X, Y_up, Z]
func to_godot(geosim_pos: Vector3) -> Vector3:
	return Vector3(geosim_pos.x, geosim_pos.z, geosim_pos.y)


## Convert Godot world position to grid pixel coordinates for heatmap.
func world_to_grid(world_pos: Vector3, grid_origin: Vector2, grid_resolution: float) -> Vector2i:
	var local_x := world_pos.x - grid_origin.x
	var local_z := world_pos.z - grid_origin.y
	return Vector2i(int(local_x / grid_resolution), int(local_z / grid_resolution))


## Convert grid pixel coordinates back to Godot world position (center of pixel).
func grid_to_world(grid_pos: Vector2i, grid_origin: Vector2, grid_resolution: float,
		y_height: float = 0.0) -> Vector3:
	return Vector3(
		grid_origin.x + (grid_pos.x + 0.5) * grid_resolution,
		y_height,
		grid_origin.y + (grid_pos.y + 0.5) * grid_resolution,
	)


## Format position as mock UTM string for display.
func format_utm(godot_pos: Vector3) -> String:
	var gs := to_geosim(godot_pos)
	return "%.1f E  %.1f N" % [gs.x + UTM_EASTING_OFFSET, gs.y + UTM_NORTHING_OFFSET]


## Format position as local GeoSim meters.
func format_local(godot_pos: Vector3) -> String:
	var gs := to_geosim(godot_pos)
	return "(%.1f, %.1f) m" % [gs.x, gs.y]
