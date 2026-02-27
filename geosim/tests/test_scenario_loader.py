"""Tests for scenario loading and ground truth management."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from geosim.scenarios.loader import (
    BuriedObject,
    Scenario,
    SoilLayer,
    Terrain,
    load_scenario,
    save_scenario,
)


@pytest.fixture
def sample_scenario_path():
    """Create a temporary scenario JSON file."""
    data = {
        "name": "Test Scenario",
        "description": "A simple test scenario",
        "earth_field": [0.0, 20e-6, 45e-6],
        "terrain": {
            "x_extent": [0.0, 10.0],
            "y_extent": [0.0, 10.0],
            "surface_elevation": 0.0,
            "layers": [
                {
                    "name": "Topsoil",
                    "z_top": 0.0,
                    "z_bottom": -0.3,
                    "conductivity": 0.05,
                }
            ],
        },
        "objects": [
            {
                "name": "Target A",
                "position": [5.0, 5.0, -1.0],
                "type": "ferrous_sphere",
                "radius": 0.05,
                "susceptibility": 1000.0,
                "conductivity": 1e6,
            },
            {
                "name": "Target B (explicit moment)",
                "position": [3.0, 3.0, -0.5],
                "type": "ferrous_sphere",
                "moment": [0.0, 0.0, 0.5],
            },
        ],
        "metadata": {"test": True},
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        return f.name


class TestLoadScenario:
    """Tests for scenario loading from JSON."""

    def test_load_basic_scenario(self, sample_scenario_path):
        """Load a basic scenario and verify fields."""
        scenario = load_scenario(sample_scenario_path)

        assert scenario.name == "Test Scenario"
        assert scenario.description == "A simple test scenario"
        assert len(scenario.objects) == 2
        assert len(scenario.terrain.layers) == 1

    def test_terrain_extents(self, sample_scenario_path):
        """Verify terrain extents are loaded correctly."""
        scenario = load_scenario(sample_scenario_path)

        assert scenario.terrain.x_extent == (0.0, 10.0)
        assert scenario.terrain.y_extent == (0.0, 10.0)

    def test_object_positions(self, sample_scenario_path):
        """Verify object positions are numpy arrays."""
        scenario = load_scenario(sample_scenario_path)

        obj_a = scenario.objects[0]
        assert isinstance(obj_a.position, np.ndarray)
        np.testing.assert_array_equal(obj_a.position, [5.0, 5.0, -1.0])

    def test_induced_moments_computed(self, sample_scenario_path):
        """Objects without explicit moments get induced moments computed."""
        scenario = load_scenario(sample_scenario_path)

        # Target A had no moment but has susceptibility + radius
        obj_a = scenario.objects[0]
        assert obj_a.moment is not None
        assert np.linalg.norm(obj_a.moment) > 0

    def test_explicit_moment_preserved(self, sample_scenario_path):
        """Objects with explicit moments are not overwritten."""
        scenario = load_scenario(sample_scenario_path)

        obj_b = scenario.objects[1]
        np.testing.assert_array_equal(obj_b.moment, [0.0, 0.0, 0.5])

    def test_magnetic_sources(self, sample_scenario_path):
        """magnetic_sources returns list of dipole source dicts."""
        scenario = load_scenario(sample_scenario_path)
        sources = scenario.magnetic_sources

        assert len(sources) == 2
        for src in sources:
            assert 'position' in src
            assert 'moment' in src

    def test_earth_field(self, sample_scenario_path):
        """Earth field is loaded as numpy array."""
        scenario = load_scenario(sample_scenario_path)

        assert isinstance(scenario.earth_field, np.ndarray)
        np.testing.assert_allclose(
            scenario.earth_field, [0.0, 20e-6, 45e-6]
        )

    def test_load_real_scenario_files(self):
        """Load each real scenario file from the scenarios directory."""
        scenarios_dir = Path(__file__).parent.parent / "scenarios"
        if not scenarios_dir.exists():
            pytest.skip("Scenarios directory not found")

        for json_file in scenarios_dir.glob("*.json"):
            scenario = load_scenario(json_file)
            assert scenario.name
            assert len(scenario.objects) > 0


class TestSaveScenario:
    """Tests for scenario serialization."""

    def test_round_trip(self, sample_scenario_path):
        """Load → save → load produces equivalent scenario."""
        scenario = load_scenario(sample_scenario_path)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            save_path = f.name

        save_scenario(scenario, save_path)
        reloaded = load_scenario(save_path)

        assert reloaded.name == scenario.name
        assert len(reloaded.objects) == len(scenario.objects)
        for orig, reloaded_obj in zip(scenario.objects, reloaded.objects):
            np.testing.assert_allclose(
                orig.position, reloaded_obj.position
            )


class TestBuriedObject:
    """Tests for BuriedObject dataclass."""

    def test_as_dipole_source(self):
        """Convert object with moment to dipole source dict."""
        obj = BuriedObject(
            name="Test",
            position=np.array([1.0, 2.0, -3.0]),
            object_type="ferrous_sphere",
            moment=np.array([0.0, 0.0, 1.0]),
        )
        source = obj.as_dipole_source()
        assert 'position' in source
        assert 'moment' in source

    def test_as_dipole_source_no_moment_raises(self):
        """Object without moment raises ValueError."""
        obj = BuriedObject(
            name="Test",
            position=np.array([1.0, 2.0, -3.0]),
            object_type="ferrous_sphere",
        )
        with pytest.raises(ValueError, match="no magnetic moment"):
            obj.as_dipole_source()
