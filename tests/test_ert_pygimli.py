"""Tests for pyGIMLi ERT backend integration.

These tests require pyGIMLi (pygimli) to be installed. They validate
that the pyGIMLi backend produces results consistent with analytical solutions.

Skip with: pytest -m "not pygimli"
"""

import numpy as np
import pytest

pygimli = pytest.importorskip("pygimli", reason="pyGIMLi not installed")

from geosim.resistivity.ert import ert_forward


pytestmark = pytest.mark.pygimli


class TestERTpyGIMLi:
    """Tests for pyGIMLi ERT forward modeling."""

    def _wenner_setup(self, a=1.0):
        """Create a simple Wenner array setup."""
        electrodes = np.array([
            [0.0, 0.0],
            [a, 0.0],
            [2 * a, 0.0],
            [3 * a, 0.0],
        ])
        measurements = [(0, 3, 1, 2)]  # C1, C2, P1, P2
        return electrodes, measurements

    def test_pygimli_backend_runs(self):
        """pyGIMLi backend completes without error."""
        electrodes, measurements = self._wenner_setup()
        result = ert_forward(
            electrode_positions=electrodes,
            measurements=measurements,
            resistivities=[100.0],
            backend="pygimli",
        )
        assert result['backend'] == 'pygimli'
        assert len(result['apparent_resistivity']) == 1

    def test_halfspace_consistency(self):
        """pyGIMLi half-space ρ_a should be close to true ρ."""
        electrodes, measurements = self._wenner_setup(a=1.0)
        rho_true = 100.0
        result = ert_forward(
            electrode_positions=electrodes,
            measurements=measurements,
            resistivities=[rho_true],
            backend="pygimli",
        )
        rho_a = result['apparent_resistivity'][0]
        # pyGIMLi numerical result should be within 20% of analytical
        assert rho_a == pytest.approx(rho_true, rel=0.2)

    def test_two_layer_contrast(self):
        """pyGIMLi detects resistivity contrast between layers."""
        electrodes, measurements = self._wenner_setup(a=0.5)

        r_cond_top = ert_forward(
            electrode_positions=electrodes,
            measurements=measurements,
            resistivities=[10.0, 1000.0],
            thicknesses=[1.0],
            backend="pygimli",
        )
        r_res_top = ert_forward(
            electrode_positions=electrodes,
            measurements=measurements,
            resistivities=[1000.0, 10.0],
            thicknesses=[1.0],
            backend="pygimli",
        )
        # Conductive top should give lower apparent resistivity
        assert r_cond_top['apparent_resistivity'][0] < r_res_top['apparent_resistivity'][0]
