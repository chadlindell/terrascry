"""3D visualization utilities using PyVista.

Provides reusable scene-building functions for geophysical simulation
visualization. All functions return PyVista objects that can be composed
into scenes and exported as static images or interactive HTML.
"""

from geosim.viz.scenes import (
    create_survey_scene,
    create_field_scene,
    render_to_image,
)
from geosim.viz.terrain import create_terrain_mesh, create_subsurface_slice
from geosim.viz.objects import create_buried_objects, create_sensor_path
from geosim.viz.fields import create_field_volume, create_gradient_surface
