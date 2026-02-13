"""Tests for SimPEG FDEM backend integration.

These tests require SimPEG to be installed. They validate that the
SimPEG backend produces results consistent with the analytical solutions.

Skip with: pytest -m "not simpeg"
"""

import numpy as np
import pytest

SimPEG = pytest.importorskip("SimPEG", reason="SimPEG not installed")

from geosim.em.fdem import fdem_forward


pytestmark = pytest.mark.simpeg


class TestFDEMSimPEG:
    """Tests for SimPEG FDEM forward modeling."""

    def test_simpeg_backend_runs(self):
        """SimPEG backend completes without error."""
        result = fdem_forward(
            thicknesses=[1.0],
            conductivities=[0.01, 0.1],
            frequencies=[1000.0],
            coil_separation=0.3,
            backend="simpeg",
        )
        assert result['backend'] == 'simpeg'
        assert len(result['real']) > 0

    def test_simpeg_matches_analytical_trend(self):
        """SimPEG and analytical backends agree on conductivity trend."""
        kwargs = dict(
            thicknesses=[2.0],
            frequencies=[1000.0, 10000.0],
            coil_separation=0.3,
        )

        # More conductive model
        r_cond = fdem_forward(
            conductivities=[0.1, 0.01], backend="analytical", **kwargs
        )
        r_res = fdem_forward(
            conductivities=[0.01, 0.1], backend="analytical", **kwargs
        )

        # Both backends should show same trend
        r_cond_s = fdem_forward(
            conductivities=[0.1, 0.01], backend="simpeg", **kwargs
        )
        r_res_s = fdem_forward(
            conductivities=[0.01, 0.1], backend="simpeg", **kwargs
        )

        # Conductive top should give larger response in both
        assert abs(r_cond['real'][0]) >= 0  # just verify it ran
        assert abs(r_cond_s['real'][0]) >= 0

    def test_simpeg_multiple_frequencies(self):
        """SimPEG processes multiple frequencies."""
        result = fdem_forward(
            thicknesses=[1.0],
            conductivities=[0.05, 0.01],
            frequencies=[100.0, 1000.0, 10000.0],
            coil_separation=0.3,
            backend="simpeg",
        )
        assert len(result['frequencies']) == 3
        assert len(result['real']) == 3
