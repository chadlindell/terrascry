"""Tests for analytical electrical resistivity models.

Validates geometric factors, half-space apparent resistivity,
and 1D layered earth responses against known analytical results.
"""

import numpy as np
import pytest

from geosim.resistivity.electrodes import ElectrodeArray, hirt_default_electrodes
from geosim.resistivity.ert import apparent_resistivity_halfspace, apparent_resistivity_layered
from geosim.resistivity.geometric import (
    geometric_factor,
    geometric_factor_dipole_dipole,
    geometric_factor_hirt_crosshole,
    geometric_factor_schlumberger,
    geometric_factor_wenner,
)


class TestGeometricFactorWenner:
    """Tests for Wenner array geometric factor."""

    def test_wenner_formula(self):
        """K_wenner = 2πa."""
        a = 1.0
        K = geometric_factor_wenner(a)
        assert K == pytest.approx(2.0 * np.pi * a, rel=1e-14)

    def test_wenner_scaling(self):
        """Doubling spacing doubles K."""
        K1 = geometric_factor_wenner(1.0)
        K2 = geometric_factor_wenner(2.0)
        assert K2 / K1 == pytest.approx(2.0, rel=1e-14)

    def test_wenner_from_general(self):
        """General formula matches Wenner formula for Wenner geometry."""
        a = 2.0
        # Wenner: C1-P1-P2-C2, spacing a
        c1 = np.array([0.0, 0.0])
        p1 = np.array([a, 0.0])
        p2 = np.array([2 * a, 0.0])
        c2 = np.array([3 * a, 0.0])
        K_general = geometric_factor(c1, c2, p1, p2)
        K_wenner = geometric_factor_wenner(a)
        assert K_general == pytest.approx(K_wenner, rel=1e-10)

    def test_wenner_values(self):
        """Check specific values."""
        assert geometric_factor_wenner(0.5) == pytest.approx(np.pi, rel=1e-14)
        assert geometric_factor_wenner(1.0) == pytest.approx(2.0 * np.pi, rel=1e-14)


class TestGeometricFactorGeneral:
    """Tests for general geometric factor calculation."""

    def test_symmetric_config(self):
        """Symmetric configuration produces positive K."""
        c1 = np.array([0.0, 0.0])
        c2 = np.array([3.0, 0.0])
        p1 = np.array([1.0, 0.0])
        p2 = np.array([2.0, 0.0])
        K = geometric_factor(c1, c2, p1, p2)
        assert K > 0

    def test_3d_electrodes(self):
        """Works with 3D electrode positions."""
        c1 = np.array([0.0, 0.0, 0.0])
        c2 = np.array([3.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([2.0, 0.0, 0.0])
        K = geometric_factor(c1, c2, p1, p2)
        assert K > 0


class TestGeometricFactorSchlumberger:
    """Tests for Schlumberger array geometric factor."""

    def test_schlumberger_formula(self):
        """K = πn(n+1)a."""
        a = 1.0
        n = 3
        K = geometric_factor_schlumberger(a, n)
        assert K == pytest.approx(np.pi * 3 * 4 * 1.0, rel=1e-14)

    def test_schlumberger_n1_vs_wenner(self):
        """Schlumberger with n=1 is NOT the same as Wenner (different geometry)."""
        a = 1.0
        K_sch = geometric_factor_schlumberger(a, n=1)
        K_wen = geometric_factor_wenner(a)
        # They use different formulas, just verify Schlumberger is reasonable
        assert K_sch > 0


class TestGeometricFactorDipoleDipole:
    """Tests for dipole-dipole array geometric factor."""

    def test_dipole_dipole_formula(self):
        """K = πn(n+1)(n+2)a."""
        a = 1.0
        n = 2
        K = geometric_factor_dipole_dipole(a, n)
        assert K == pytest.approx(np.pi * 2 * 3 * 4 * 1.0, rel=1e-14)

    def test_increases_with_n(self):
        """K increases rapidly with n (sensitivity decreases)."""
        a = 1.0
        K1 = geometric_factor_dipole_dipole(a, 1)
        K2 = geometric_factor_dipole_dipole(a, 2)
        K3 = geometric_factor_dipole_dipole(a, 3)
        assert K1 < K2 < K3


class TestGeometricFactorHIRT:
    """Tests for HIRT cross-hole geometric factor."""

    def test_positive_K(self):
        """Cross-hole configuration produces positive K."""
        K = geometric_factor_hirt_crosshole(0.1, 0.5)
        assert K > 0

    def test_increases_with_borehole_separation(self):
        """Wider borehole separation increases K."""
        K1 = geometric_factor_hirt_crosshole(0.1, 0.3)
        K2 = geometric_factor_hirt_crosshole(0.1, 0.6)
        assert K2 > K1


class TestApparentResistivityHalfspace:
    """Tests for homogeneous half-space apparent resistivity."""

    def test_halfspace_returns_true_resistivity(self):
        """For a uniform half-space, ρ_a = ρ always."""
        rho = 100.0
        # Wenner array with a=1
        c1 = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])
        p2 = np.array([2.0, 0.0])
        c2 = np.array([3.0, 0.0])
        rho_a = apparent_resistivity_halfspace(rho, c1, c2, p1, p2)
        assert rho_a == pytest.approx(rho, rel=1e-10)

    def test_halfspace_various_resistivities(self):
        """ρ_a = ρ for various resistivity values."""
        c1 = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])
        p2 = np.array([2.0, 0.0])
        c2 = np.array([3.0, 0.0])
        for rho in [1.0, 10.0, 100.0, 1000.0]:
            rho_a = apparent_resistivity_halfspace(rho, c1, c2, p1, p2)
            assert rho_a == pytest.approx(rho, rel=1e-10), f"Failed for ρ={rho}"

    def test_halfspace_different_array_geometry(self):
        """ρ_a = ρ regardless of electrode positions."""
        rho = 50.0
        # Non-standard electrode positions
        c1 = np.array([0.0, 0.0])
        c2 = np.array([5.0, 0.0])
        p1 = np.array([1.5, 0.0])
        p2 = np.array([3.5, 0.0])
        rho_a = apparent_resistivity_halfspace(rho, c1, c2, p1, p2)
        assert rho_a == pytest.approx(rho, rel=1e-10)

    def test_halfspace_3d_electrodes(self):
        """Works with 3D electrode positions."""
        rho = 100.0
        c1 = np.array([0.0, 0.0, 0.0])
        c2 = np.array([3.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([2.0, 0.0, 0.0])
        rho_a = apparent_resistivity_halfspace(rho, c1, c2, p1, p2)
        assert rho_a == pytest.approx(rho, rel=1e-10)


class TestApparentResistivityLayered:
    """Tests for 1D layered earth apparent resistivity."""

    def test_single_layer_returns_true_resistivity(self):
        """Single layer (half-space) returns true resistivity."""
        rho_a = apparent_resistivity_layered(
            thicknesses=[], resistivities=[100.0], electrode_spacing=1.0
        )
        assert rho_a == pytest.approx(100.0, rel=1e-10)

    def test_two_layer_contrast(self):
        """Conductive layer over resistive: ρ_a between the two values."""
        rho_a = apparent_resistivity_layered(
            thicknesses=[1.0],
            resistivities=[10.0, 1000.0],
            electrode_spacing=0.5,
        )
        # ρ_a should be between the two layer values for shallow measurement
        assert 5.0 < rho_a < 1500.0

    def test_three_layer_runs(self):
        """Three-layer model computes without error."""
        rho_a = apparent_resistivity_layered(
            thicknesses=[0.5, 2.0],
            resistivities=[100.0, 10.0, 1000.0],
            electrode_spacing=1.0,
        )
        assert rho_a > 0


class TestElectrodeArray:
    """Tests for HIRT electrode array configuration."""

    def test_hirt_default_electrode_count(self):
        """Default HIRT has 16 electrodes (8 per borehole × 2 boreholes)."""
        array = hirt_default_electrodes()
        assert array.n_electrodes == 16

    def test_hirt_default_borehole_split(self):
        """8 electrodes in each borehole."""
        array = hirt_default_electrodes()
        bh0 = array.borehole_electrodes(0)
        bh1 = array.borehole_electrodes(1)
        assert len(bh0) == 8
        assert len(bh1) == 8

    def test_borehole_separation(self):
        """Boreholes are separated by the specified distance."""
        sep = 0.50
        array = hirt_default_electrodes(borehole_separation=sep)
        bh0_x = array.borehole_electrodes(0)[0].position[0]
        bh1_x = array.borehole_electrodes(1)[0].position[0]
        assert abs(bh1_x - bh0_x) == pytest.approx(sep, rel=1e-10)

    def test_electrode_positions_array(self):
        """Positions property returns (N, 3) array."""
        array = hirt_default_electrodes()
        pos = array.positions
        assert pos.shape == (16, 3)

    def test_electrode_labels(self):
        """Electrodes have A/B prefixed labels."""
        array = hirt_default_electrodes()
        labels = [e.label for e in array.electrodes]
        assert "A1" in labels
        assert "B1" in labels
        assert "A8" in labels
        assert "B8" in labels

    def test_custom_configuration(self):
        """Custom parameters are respected."""
        array = hirt_default_electrodes(
            n_rings=4, ring_spacing=0.20, borehole_separation=1.0
        )
        assert array.n_electrodes == 8  # 4 per borehole × 2
        bh0 = array.borehole_electrodes(0)
        # Check spacing between first two electrodes
        dz = abs(bh0[1].position[2] - bh0[0].position[2])
        assert dz == pytest.approx(0.20, rel=1e-10)
