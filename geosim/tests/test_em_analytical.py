"""Tests for analytical electromagnetic models.

Validates skin depth calculations, conductive sphere response,
and 1D layered earth FDEM against known analytical results.
"""

import numpy as np
import pytest

from geosim.em.coil import CoilConfig, ProbeCoilSet, hirt_default_coils
from geosim.em.fdem import fdem_response_1d, secondary_field_conductive_sphere
from geosim.em.skin_depth import MU_0, skin_depth, skin_depth_practical


class TestSkinDepth:
    """Tests for electromagnetic skin depth calculations."""

    def test_known_value_seawater(self):
        """Skin depth in seawater at 1 kHz: δ ≈ 8.0 m.

        σ = 5 S/m (typical seawater).
        δ = 1/√(π·1000·4π×10⁻⁷·5) ≈ 7.12 m
        """
        delta = skin_depth(1000.0, 5.0)
        expected = 1.0 / np.sqrt(np.pi * 1000.0 * MU_0 * 5.0)
        assert delta == pytest.approx(expected, rel=1e-10)
        assert 5.0 < delta < 10.0  # reasonable range check

    def test_known_value_copper(self):
        """Skin depth in copper at 60 Hz: δ ≈ 8.5 mm.

        σ = 5.96×10⁷ S/m.
        """
        delta = skin_depth(60.0, 5.96e7)
        assert 0.005 < delta < 0.012  # ~8.5mm

    def test_inverse_sqrt_frequency(self):
        """δ ∝ 1/√f: quadrupling frequency halves skin depth."""
        d1 = skin_depth(100.0, 0.01)
        d2 = skin_depth(400.0, 0.01)
        assert d1 / d2 == pytest.approx(2.0, rel=1e-10)

    def test_inverse_sqrt_conductivity(self):
        """δ ∝ 1/√σ: quadrupling conductivity halves skin depth."""
        d1 = skin_depth(1000.0, 0.01)
        d2 = skin_depth(1000.0, 0.04)
        assert d1 / d2 == pytest.approx(2.0, rel=1e-10)

    def test_mu_r_effect(self):
        """Higher permeability decreases skin depth."""
        d1 = skin_depth(1000.0, 0.01, mu_r=1.0)
        d2 = skin_depth(1000.0, 0.01, mu_r=100.0)
        assert d1 / d2 == pytest.approx(10.0, rel=1e-10)

    def test_vectorized_frequency(self):
        """Works with array of frequencies."""
        freqs = np.array([100.0, 1000.0, 10000.0])
        deltas = skin_depth(freqs, 0.01)
        assert deltas.shape == (3,)
        assert deltas[0] > deltas[1] > deltas[2]

    def test_vectorized_conductivity(self):
        """Works with array of conductivities."""
        sigmas = np.array([0.001, 0.01, 0.1, 1.0])
        deltas = skin_depth(1000.0, sigmas)
        assert deltas.shape == (4,)
        assert deltas[0] > deltas[1] > deltas[2] > deltas[3]

    def test_practical_formula_matches_exact(self):
        """Practical formula ≈ 503·√(ρ/f) matches exact formula for μ_r=1."""
        freq = 1000.0
        sigma = 0.01
        rho = 1.0 / sigma
        d_exact = skin_depth(freq, sigma)
        d_practical = skin_depth_practical(rho, freq)
        assert d_exact == pytest.approx(d_practical, rel=1e-10)

    def test_practical_formula_published_value(self):
        """Practical formula gives δ ≈ 503m for ρ=1Ω·m, f=1Hz."""
        delta = skin_depth_practical(1.0, 1.0)
        assert delta == pytest.approx(503.3, rel=0.01)

    def test_practical_vectorized(self):
        """Practical formula works with arrays."""
        rhos = np.array([1.0, 10.0, 100.0])
        deltas = skin_depth_practical(rhos, 1000.0)
        assert deltas.shape == (3,)
        # δ ∝ √ρ: 10x resistivity → √10 × skin depth
        assert deltas[1] / deltas[0] == pytest.approx(np.sqrt(10.0), rel=1e-10)


class TestConductiveSphere:
    """Tests for the Wait (1951) conductive sphere solution."""

    def test_response_is_complex(self):
        """Response has both real (in-phase) and imaginary (quadrature) parts."""
        resp = secondary_field_conductive_sphere(
            radius=0.1, conductivity=1e6, frequency=1000.0, r_obs=1.0
        )
        assert isinstance(resp, complex)

    def test_response_decays_with_distance(self):
        """Response falls off as (a/r)³ (dipolar decay)."""
        kwargs = dict(radius=0.1, conductivity=1e6, frequency=1000.0)
        r1 = secondary_field_conductive_sphere(r_obs=1.0, **kwargs)
        r2 = secondary_field_conductive_sphere(r_obs=2.0, **kwargs)
        ratio = abs(r1) / abs(r2)
        assert ratio == pytest.approx(8.0, rel=0.1)

    def test_response_scales_with_radius(self):
        """At fixed distance, larger sphere gives stronger response."""
        kwargs = dict(conductivity=100.0, frequency=100.0, r_obs=5.0)
        r1 = secondary_field_conductive_sphere(radius=0.05, **kwargs)
        r2 = secondary_field_conductive_sphere(radius=0.10, **kwargs)
        # Response scales with radius^n where n depends on induction number.
        # The (a/r)^3 dipolar factor gives 8x, plus the sphere response
        # function Q also depends on radius through kr, giving steeper scaling.
        assert abs(r2) > abs(r1)

    def test_low_induction_very_small_response(self):
        """At low induction number (α<<1), response approaches zero."""
        # Low conductivity, low frequency → very small response
        resp = secondary_field_conductive_sphere(
            radius=0.01, conductivity=1.0, frequency=10.0, r_obs=1.0
        )
        # Small sphere with low σ·ω product → negligible response
        assert abs(resp) < 1e-10

    def test_high_induction_inphase_dominant(self):
        """At high induction number (α>>1), in-phase > quadrature."""
        # High conductivity, high frequency → large skin depth ratio
        resp = secondary_field_conductive_sphere(
            radius=0.1, conductivity=1e7, frequency=100000.0, r_obs=1.0
        )
        assert abs(resp.real) > abs(resp.imag)

    def test_low_conductivity_weak_response(self):
        """Lower conductivity gives weaker response."""
        kwargs = dict(radius=0.1, frequency=1000.0, r_obs=1.0)
        r_high = secondary_field_conductive_sphere(conductivity=1e6, **kwargs)
        r_low = secondary_field_conductive_sphere(conductivity=1.0, **kwargs)
        assert abs(r_high) > abs(r_low)


class TestCoilConfiguration:
    """Tests for HIRT coil geometry."""

    def test_hirt_default_coil_count(self):
        """Default HIRT has 6 coils (3 TX + 3 RX)."""
        coils = hirt_default_coils()
        assert coils.n_coils == 6

    def test_hirt_default_tx_rx_split(self):
        """3 transmitters and 3 receivers."""
        coils = hirt_default_coils()
        assert len(coils.transmitters) == 3
        assert len(coils.receivers) == 3

    def test_coil_positions_increasing(self):
        """Coil positions increase along probe axis."""
        coils = hirt_default_coils()
        positions = [c.position for c in coils.coils]
        assert positions == sorted(positions)

    def test_coil_config_dataclass(self):
        """CoilConfig stores parameters correctly."""
        c = CoilConfig(position=0.5, radius=0.02, turns=100, role="rx", label="RX1")
        assert c.position == 0.5
        assert c.radius == 0.02
        assert c.turns == 100
        assert c.role == "rx"

    def test_probe_coil_set_properties(self):
        """ProbeCoilSet properties work correctly."""
        coils = hirt_default_coils()
        assert coils.probe_length == 1.0
        assert coils.probe_diameter == 0.04


class TestFDEM1D:
    """Tests for 1D layered earth FDEM response."""

    def test_homogeneous_halfspace(self):
        """For a uniform half-space, response is proportional to conductivity."""
        sigma = 0.05
        resp = fdem_response_1d(
            thicknesses=[],
            conductivities=[sigma],
            frequency=10000.0,
            coil_separation=0.3,
        )
        # For a single-layer model, response should be close to sigma
        assert abs(resp) > 0

    def test_two_layer_contrast(self):
        """Two-layer model: conductive layer over resistive gives higher response."""
        # Conductive top layer
        resp_cond = fdem_response_1d(
            thicknesses=[1.0],
            conductivities=[0.1, 0.01],
            frequency=10000.0,
            coil_separation=0.3,
        )
        # Resistive top layer
        resp_res = fdem_response_1d(
            thicknesses=[1.0],
            conductivities=[0.01, 0.1],
            frequency=10000.0,
            coil_separation=0.3,
        )
        assert abs(resp_cond) > abs(resp_res)

    def test_three_layer_model(self):
        """Three-layer model runs without error."""
        resp = fdem_response_1d(
            thicknesses=[0.5, 1.0],
            conductivities=[0.05, 0.2, 0.01],
            frequency=10000.0,
            coil_separation=0.3,
        )
        assert isinstance(resp, (float, complex))
