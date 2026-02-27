"""Tests for geosim.viz module (PyVista 3D visualization)."""

from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
pv.OFF_SCREEN = True

from geosim.scenarios.loader import load_scenario
from geosim.viz.terrain import create_terrain_mesh, create_subsurface_slice, create_soil_layers
from geosim.viz.objects import (
    create_buried_objects, create_sensor_path,
    create_sensor_positions, create_moment_arrows,
)
from geosim.viz.fields import (
    create_field_volume, create_gradient_surface,
    create_field_arrows, create_isosurfaces,
)
from geosim.viz.scenes import create_survey_scene, create_field_scene, render_to_image

SCENARIO_PATH = "scenarios/single-ferrous-target.json"


@pytest.fixture
def scenario():
    return load_scenario(SCENARIO_PATH)


@pytest.fixture
def sources(scenario):
    return scenario.magnetic_sources


class TestTerrain:
    def test_terrain_mesh_is_structured_grid(self, scenario):
        mesh = create_terrain_mesh(scenario)
        assert isinstance(mesh, pv.StructuredGrid)
        assert mesh.n_points > 0

    def test_terrain_has_elevation(self, scenario):
        mesh = create_terrain_mesh(scenario)
        assert "elevation" in mesh.point_data

    def test_subsurface_slice(self, scenario):
        mesh = create_subsurface_slice(scenario, depth=3.0, resolution=20)
        assert isinstance(mesh, pv.StructuredGrid)
        assert "conductivity" in mesh.point_data

    def test_soil_layers(self, scenario):
        layers = create_soil_layers(scenario)
        assert isinstance(layers, list)
        # May be empty if scenario has no layers defined
        for layer in layers:
            assert isinstance(layer, pv.PolyData)


class TestObjects:
    def test_buried_objects_returns_list(self, scenario):
        objects = create_buried_objects(scenario, scale_factor=2.0)
        assert isinstance(objects, list)
        assert len(objects) > 0

    def test_buried_object_has_properties(self, scenario):
        objects = create_buried_objects(scenario)
        mesh, props = objects[0]
        assert isinstance(mesh, pv.PolyData)
        assert "name" in props
        assert "depth" in props
        assert "moment" in props

    def test_sensor_path_tube(self):
        positions = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
        tube = create_sensor_path(positions, height=0.2)
        assert isinstance(tube, pv.PolyData)
        assert tube.n_points > 0

    def test_sensor_positions(self):
        positions = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
        values = np.array([10.0, 20.0, 30.0])
        cloud = create_sensor_positions(positions, values, height=0.2)
        assert isinstance(cloud, pv.PolyData)
        assert cloud.n_points == 3
        assert "gradient" in cloud.point_data

    def test_moment_arrows(self, scenario):
        arrows = create_moment_arrows(scenario)
        assert isinstance(arrows, pv.PolyData)


class TestFields:
    def test_field_volume(self, sources):
        vol = create_field_volume(sources, bounds=(-2, 2, -2, 2, -2, 2), resolution=8)
        assert isinstance(vol, pv.ImageData)
        assert "field" in vol.point_data
        assert "B_vector" in vol.point_data

    def test_gradient_surface(self, sources):
        surf = create_gradient_surface(sources, (-3, 3), (-3, 3), resolution=10)
        assert isinstance(surf, pv.StructuredGrid)
        assert "gradient_nT" in surf.point_data

    def test_field_arrows(self, sources):
        arrows = create_field_arrows(sources, (-2, 2, -2, 2, -2, 2), resolution=4)
        assert isinstance(arrows, pv.PolyData)

    def test_isosurfaces(self, sources):
        vol = create_field_volume(sources, (-2, 2, -2, 2, -2, 2), resolution=10)
        surfaces = create_isosurfaces(vol, n_levels=3)
        assert isinstance(surfaces, list)
        # Might be empty if field is too uniform, but shouldn't be for a dipole
        for s in surfaces:
            assert isinstance(s, pv.PolyData)


class TestScenes:
    def test_survey_scene(self, scenario):
        plotter = create_survey_scene(scenario, off_screen=True)
        assert isinstance(plotter, pv.Plotter)

    def test_survey_scene_with_positions(self, scenario):
        positions = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
        gradient = np.array([0.0, 5.0, 10.0, 3.0])
        plotter = create_survey_scene(
            scenario, positions=positions, gradient_data=gradient, off_screen=True,
        )
        assert isinstance(plotter, pv.Plotter)

    def test_field_scene(self, sources):
        plotter = create_field_scene(sources, off_screen=True, n_isosurfaces=3)
        assert isinstance(plotter, pv.Plotter)

    def test_render_to_image(self, scenario, tmp_path):
        plotter = create_survey_scene(scenario, off_screen=True)
        out = tmp_path / "test.png"
        result = render_to_image(plotter, out)
        assert result.exists()
        assert result.stat().st_size > 0
