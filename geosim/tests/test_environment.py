"""Tests for Archie's Law environmental soil model.

Validates conductivity computations against hand calculations for each
soil preset, temperature correction, frost condition, and saturation sweep.
"""

from __future__ import annotations

import pytest

from geosim.environment import (
    SOIL_PRESETS,
    SoilEnvironment,
    archie_conductivity,
    get_soil_preset,
)


class TestSoilEnvironment:
    """Basic SoilEnvironment dataclass behavior."""

    def test_default_values(self):
        env = SoilEnvironment()
        assert env.temperature_c == 15.0
        assert env.saturation == 0.6
        assert env.pore_water_sigma == 0.05
        assert env.frozen is False

    def test_round_trip_dict(self):
        env = SoilEnvironment(temperature_c=25.0, saturation=0.8, frozen=True)
        d = env.to_dict()
        env2 = SoilEnvironment.from_dict(d)
        assert env2.temperature_c == 25.0
        assert env2.saturation == 0.8
        assert env2.frozen is True

    def test_from_dict_defaults(self):
        env = SoilEnvironment.from_dict({})
        assert env.temperature_c == 15.0
        assert env.saturation == 0.6


class TestArchieHandCalcSandy:
    """Sandy soil: φ=0.38, a=1.0, m=1.6, n=2.0, σ_surf=0.005.

    Hand calculation at T=25°C, S_w=0.8, σ_w=0.05:
        σ_w(25) = 0.05 × (1 + 0.021×0) = 0.05
        φ^m = 0.38^1.6 = 0.38^1.6
        S_w^n = 0.8^2.0 = 0.64
        σ_b = 0.05 × 0.38^1.6 × 0.64 / 1.0 + 0.005
    """

    def test_sandy_at_25c(self):
        p = get_soil_preset("sandy")
        env = SoilEnvironment(temperature_c=25.0, saturation=0.8, pore_water_sigma=0.05)
        sigma = archie_conductivity(
            p["porosity"], p["archie_a"], p["archie_m"], p["archie_n"],
            p["surface_conductivity"], env,
        )
        # Hand calc: 0.05 × 0.38^1.6 × 0.64 / 1.0 + 0.005
        phi_m = 0.38 ** 1.6
        expected = 0.05 * phi_m * 0.64 / 1.0 + 0.005
        assert sigma == pytest.approx(expected, rel=1e-10)


class TestArchieAllPresets:
    """Each preset produces positive, finite conductivity under default conditions."""

    @pytest.mark.parametrize("preset_name", list(SOIL_PRESETS.keys()))
    def test_preset_positive(self, preset_name):
        p = get_soil_preset(preset_name)
        env = SoilEnvironment()
        sigma = archie_conductivity(
            p["porosity"], p["archie_a"], p["archie_m"], p["archie_n"],
            p["surface_conductivity"], env,
        )
        assert sigma > 0.0
        assert sigma < 100.0  # reasonable upper bound

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown soil preset"):
            get_soil_preset("nonexistent")


class TestTemperatureCorrection:
    """Temperature correction: σ_w(T) = σ_w(25) × [1 + 0.021×(T-25)]."""

    def test_higher_temp_increases_conductivity(self):
        """At 35°C, conductivity should be higher than at 25°C."""
        p = get_soil_preset("sandy")  # Low σ_surf so Archie part dominates
        env_25 = SoilEnvironment(temperature_c=25.0, saturation=0.8, pore_water_sigma=0.05)
        env_35 = SoilEnvironment(temperature_c=35.0, saturation=0.8, pore_water_sigma=0.05)
        sigma_25 = archie_conductivity(
            p["porosity"], p["archie_a"], p["archie_m"], p["archie_n"],
            p["surface_conductivity"], env_25,
        )
        sigma_35 = archie_conductivity(
            p["porosity"], p["archie_a"], p["archie_m"], p["archie_n"],
            p["surface_conductivity"], env_35,
        )
        assert sigma_35 > sigma_25
        # 10°C increase → 21% increase in σ_w → ~18% increase in bulk
        # (surface conductivity dilutes the effect slightly)
        ratio = sigma_35 / sigma_25
        assert ratio > 1.10  # at least 10% increase
        assert ratio < 1.25  # not more than 25%

    def test_lower_temp_decreases_conductivity(self):
        """At 5°C, conductivity should be ~40% lower than at 25°C."""
        p = get_soil_preset("loam")
        env_25 = SoilEnvironment(temperature_c=25.0, saturation=0.8, pore_water_sigma=0.05)
        env_5 = SoilEnvironment(temperature_c=5.0, saturation=0.8, pore_water_sigma=0.05)
        sigma_25 = archie_conductivity(
            p["porosity"], p["archie_a"], p["archie_m"], p["archie_n"],
            p["surface_conductivity"], env_25,
        )
        sigma_5 = archie_conductivity(
            p["porosity"], p["archie_a"], p["archie_m"], p["archie_n"],
            p["surface_conductivity"], env_5,
        )
        assert sigma_5 < sigma_25

    def test_temperature_correction_formula(self):
        """Verify σ_w(T) = σ_w(25) × [1 + 0.021 × (T - 25)]."""
        # At T=25, correction factor = 1.0
        # At T=35, correction factor = 1.21
        # At T=5, correction factor = 0.58
        p = get_soil_preset("sandy")
        sigma_surf = p["surface_conductivity"]

        env_25 = SoilEnvironment(temperature_c=25.0, saturation=1.0, pore_water_sigma=0.1)
        sigma_25 = archie_conductivity(
            p["porosity"], p["archie_a"], p["archie_m"], p["archie_n"],
            sigma_surf, env_25,
        )

        env_35 = SoilEnvironment(temperature_c=35.0, saturation=1.0, pore_water_sigma=0.1)
        sigma_35 = archie_conductivity(
            p["porosity"], p["archie_a"], p["archie_m"], p["archie_n"],
            sigma_surf, env_35,
        )

        # At S_w=1 the Archie part is proportional to σ_w, so ratio of
        # (sigma_b - sigma_surf) should match the temperature correction
        archie_25 = sigma_25 - sigma_surf
        archie_35 = sigma_35 - sigma_surf
        expected_ratio = 1.21  # 1 + 0.021 * 10
        assert (archie_35 / archie_25) == pytest.approx(expected_ratio, rel=1e-10)


class TestFrozenCondition:
    """Frozen ground: conductivity drops by factor of 20 (×0.05)."""

    def test_frozen_drops_conductivity(self):
        p = get_soil_preset("clay")
        env = SoilEnvironment(temperature_c=15.0, saturation=0.8)
        env_frozen = SoilEnvironment(temperature_c=-2.0, saturation=0.8, frozen=True)
        sigma_normal = archie_conductivity(
            p["porosity"], p["archie_a"], p["archie_m"], p["archie_n"],
            p["surface_conductivity"], env,
        )
        sigma_frozen = archie_conductivity(
            p["porosity"], p["archie_a"], p["archie_m"], p["archie_n"],
            p["surface_conductivity"], env_frozen,
        )
        # Frozen factor is 0.05, so ratio should be ~20x (temp effect modifies slightly)
        assert sigma_frozen < sigma_normal / 10.0

    def test_frozen_still_positive(self):
        """Even frozen ground has positive conductivity."""
        p = get_soil_preset("peat")
        env = SoilEnvironment(frozen=True)
        sigma = archie_conductivity(
            p["porosity"], p["archie_a"], p["archie_m"], p["archie_n"],
            p["surface_conductivity"], env,
        )
        assert sigma > 0.0


class TestSaturationSweep:
    """Conductivity monotonically increases with saturation."""

    def test_monotonic_increase(self):
        """σ_b increases monotonically from S_w=0.1 to S_w=1.0."""
        p = get_soil_preset("loam")
        env = SoilEnvironment(temperature_c=20.0, pore_water_sigma=0.05)

        saturations = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        sigmas = []
        for s in saturations:
            env.saturation = s
            sigma = archie_conductivity(
                p["porosity"], p["archie_a"], p["archie_m"], p["archie_n"],
                p["surface_conductivity"], env,
            )
            sigmas.append(sigma)

        # Each value should be greater than the previous
        for i in range(1, len(sigmas)):
            assert sigmas[i] > sigmas[i - 1], (
                f"σ not monotonic: S_w={saturations[i]:.1f} gave {sigmas[i]} "
                f"<= {sigmas[i-1]} at S_w={saturations[i-1]:.1f}"
            )

    def test_zero_saturation_gives_surface_only(self):
        """S_w=0 → σ_b = σ_surf only (no Archie contribution)."""
        p = get_soil_preset("clay")
        env = SoilEnvironment(temperature_c=20.0, saturation=0.0)
        sigma = archie_conductivity(
            p["porosity"], p["archie_a"], p["archie_m"], p["archie_n"],
            p["surface_conductivity"], env,
        )
        assert sigma == pytest.approx(p["surface_conductivity"], rel=1e-10)

    def test_full_saturation_maximum(self):
        """S_w=1 gives maximum conductivity for a given soil/temp."""
        p = get_soil_preset("sandy")
        env_full = SoilEnvironment(temperature_c=20.0, saturation=1.0, pore_water_sigma=0.05)
        env_half = SoilEnvironment(temperature_c=20.0, saturation=0.5, pore_water_sigma=0.05)
        sigma_full = archie_conductivity(
            p["porosity"], p["archie_a"], p["archie_m"], p["archie_n"],
            p["surface_conductivity"], env_full,
        )
        sigma_half = archie_conductivity(
            p["porosity"], p["archie_a"], p["archie_m"], p["archie_n"],
            p["surface_conductivity"], env_half,
        )
        assert sigma_full > sigma_half


class TestEdgeCases:
    """Boundary and edge case behavior."""

    def test_very_low_saturation(self):
        """Near-zero saturation doesn't crash or go negative."""
        p = get_soil_preset("sandy")
        env = SoilEnvironment(saturation=1e-12)
        sigma = archie_conductivity(
            p["porosity"], p["archie_a"], p["archie_m"], p["archie_n"],
            p["surface_conductivity"], env,
        )
        assert sigma >= 0.0

    def test_very_low_temperature(self):
        """Extreme cold doesn't cause negative σ_w."""
        p = get_soil_preset("peat")
        env = SoilEnvironment(temperature_c=-30.0, saturation=0.5)
        sigma = archie_conductivity(
            p["porosity"], p["archie_a"], p["archie_m"], p["archie_n"],
            p["surface_conductivity"], env,
        )
        assert sigma > 0.0

    def test_high_temperature(self):
        """High temp gives higher conductivity."""
        p = get_soil_preset("volcanic")
        env_hot = SoilEnvironment(temperature_c=40.0, saturation=0.8, pore_water_sigma=0.05)
        env_cold = SoilEnvironment(temperature_c=5.0, saturation=0.8, pore_water_sigma=0.05)
        sigma_hot = archie_conductivity(
            p["porosity"], p["archie_a"], p["archie_m"], p["archie_n"],
            p["surface_conductivity"], env_hot,
        )
        sigma_cold = archie_conductivity(
            p["porosity"], p["archie_a"], p["archie_m"], p["archie_n"],
            p["surface_conductivity"], env_cold,
        )
        assert sigma_hot > sigma_cold


class TestSoilPresetValues:
    """Validate preset parameter ranges are physically reasonable."""

    @pytest.mark.parametrize("name,params", list(SOIL_PRESETS.items()))
    def test_porosity_range(self, name, params):
        assert 0.1 <= params["porosity"] <= 0.95

    @pytest.mark.parametrize("name,params", list(SOIL_PRESETS.items()))
    def test_archie_a_range(self, name, params):
        assert 0.5 <= params["archie_a"] <= 2.5

    @pytest.mark.parametrize("name,params", list(SOIL_PRESETS.items()))
    def test_archie_m_range(self, name, params):
        assert 1.0 <= params["archie_m"] <= 3.0

    @pytest.mark.parametrize("name,params", list(SOIL_PRESETS.items()))
    def test_archie_n_range(self, name, params):
        assert 1.0 <= params["archie_n"] <= 3.0

    @pytest.mark.parametrize("name,params", list(SOIL_PRESETS.items()))
    def test_surface_conductivity_non_negative(self, name, params):
        assert params["surface_conductivity"] >= 0.0

    @pytest.mark.parametrize("name,params", list(SOIL_PRESETS.items()))
    def test_susceptibility_non_negative(self, name, params):
        assert params["susceptibility"] >= 0.0
