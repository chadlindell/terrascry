"""Tests for HIRT borehole survey pipeline.

Validates the full HIRT measurement chain: coordinate transforms,
measurement generation, FDEM/ERT simulation, CSV export, and the
top-level run_hirt_survey() entry point.

Primary test scenario: swamp-crash-site.json (has hirt_config).
"""

from pathlib import Path

import numpy as np
import pytest

from geosim.em.coil import hirt_default_coils
from geosim.scenarios.loader import AnomalyZone, ProbeConfig, load_scenario
from geosim.sensors.hirt import (
    HIRTSurveyConfig,
    export_ert_csv,
    export_fdem_csv,
    export_hirt_ert_csv,
    export_hirt_mit_csv,
    generate_ert_measurements,
    generate_fdem_measurements,
    probe_to_global,
    run_hirt_survey,
    simulate_ert,
    simulate_fdem,
)

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"
SWAMP_SCENARIO = SCENARIOS_DIR / "swamp-crash-site.json"
SINGLE_TARGET = SCENARIOS_DIR / "single-ferrous-target.json"

# Default coil set: 3 TX + 3 RX = 9 pairs
DEFAULT_COILS = hirt_default_coils()
DEFAULT_FREQUENCIES = [1000.0, 5000.0, 25000.0]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_probes():
    """Two vertical probes matching swamp-crash-site.json."""
    return [
        ProbeConfig(
            position=[11.0, 12.5, 0.0], length=1.0,
            orientation="vertical",
            ring_depths=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            coil_depths=[0.1, 0.25, 0.4, 0.55, 0.7, 0.85],
        ),
        ProbeConfig(
            position=[14.0, 12.5, 0.0], length=1.0,
            orientation="vertical",
            ring_depths=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            coil_depths=[0.1, 0.25, 0.4, 0.55, 0.7, 0.85],
        ),
    ]


@pytest.fixture
def single_probe():
    """Single vertical probe."""
    return [
        ProbeConfig(
            position=[0.0, 0.0, 0.0], length=1.0,
            orientation="vertical",
            ring_depths=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            coil_depths=[0.1, 0.25, 0.4, 0.55, 0.7, 0.85],
        ),
    ]


@pytest.fixture
def swamp_scenario():
    """Loaded swamp-crash-site scenario."""
    return load_scenario(SWAMP_SCENARIO)


@pytest.fixture
def fdem_measurements(two_probes):
    """FDEM measurement sequence for 2 probes."""
    return generate_fdem_measurements(
        two_probes, DEFAULT_COILS, DEFAULT_FREQUENCIES,
    )


@pytest.fixture
def ert_measurements(two_probes):
    """ERT measurement sequence for 2 probes."""
    return generate_ert_measurements(two_probes)


@pytest.fixture
def simple_cond_model():
    """Homogeneous conductivity model."""
    return {'thicknesses': [], 'conductivities': [0.01]}


@pytest.fixture
def simple_resistivity_model():
    """Homogeneous resistivity model."""
    return {'thicknesses': [], 'resistivities': [100.0]}


# ===========================================================================
# TestProbeToGlobal
# ===========================================================================

class TestProbeToGlobal:
    """Coordinate transform: probe-local depth → global [x, y, z]."""

    def test_vertical_probe_at_origin(self):
        """Probe at origin, depth 0.5 → [0, 0, -0.5]."""
        probe = ProbeConfig(position=[0.0, 0.0, 0.0], length=1.0)
        result = probe_to_global(0.5, probe)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, -0.5])

    def test_offset_probe(self):
        """Probe at [10, 5, 0], depth 0.3 → [10, 5, -0.3]."""
        probe = ProbeConfig(position=[10.0, 5.0, 0.0], length=1.0)
        result = probe_to_global(0.3, probe)
        np.testing.assert_array_almost_equal(result, [10.0, 5.0, -0.3])

    def test_vectorized(self):
        """Multiple depths at once → (N, 3) array."""
        probe = ProbeConfig(position=[1.0, 2.0, 0.0], length=1.0)
        depths = np.array([0.1, 0.5, 0.9])
        result = probe_to_global(depths, probe)
        assert result.shape == (3, 3)
        np.testing.assert_array_almost_equal(result[:, 0], [1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result[:, 1], [2.0, 2.0, 2.0])
        np.testing.assert_array_almost_equal(result[:, 2], [-0.1, -0.5, -0.9])

    def test_top_equals_surface(self):
        """Depth 0 = probe top = surface position."""
        probe = ProbeConfig(position=[5.0, 3.0, 0.0], length=1.0)
        result = probe_to_global(0.0, probe)
        np.testing.assert_array_almost_equal(result, [5.0, 3.0, 0.0])

    def test_angled_probe_has_lateral_offset(self):
        """Angled probe maps deeper points with lateral displacement."""
        probe = ProbeConfig(
            position=[0.0, 0.0, 0.0],
            length=3.0,
            orientation="angled",
            tilt_deg=10.0,
            azimuth_deg=90.0,  # East
        )
        result = probe_to_global(2.0, probe)
        assert result[0] > 0.0
        assert result[2] < 0.0


# ===========================================================================
# TestFDEMMeasurementGeneration
# ===========================================================================

class TestFDEMMeasurementGeneration:
    """Validates generate_fdem_measurements."""

    def test_correct_count_two_probes(self, two_probes):
        """2 probes × 9 TX-RX pairs × 3 freq = 54 intra + 54 cross = 108."""
        meas = generate_fdem_measurements(
            two_probes, DEFAULT_COILS, DEFAULT_FREQUENCIES,
        )
        assert len(meas) == 108

    def test_positive_separations(self, fdem_measurements):
        """All coil separations are positive."""
        for m in fdem_measurements:
            assert m.coil_separation > 0

    def test_frequencies_match(self, fdem_measurements):
        """All frequencies come from the configured list."""
        for m in fdem_measurements:
            assert m.frequency in DEFAULT_FREQUENCIES

    def test_no_tx_equals_rx(self, fdem_measurements):
        """TX and RX are always different coils."""
        for m in fdem_measurements:
            assert m.tx_index != m.rx_index

    def test_single_probe_count(self, single_probe):
        """1 probe: 9 TX-RX pairs × 3 freq = 27 (intra only, no cross)."""
        meas = generate_fdem_measurements(
            single_probe, DEFAULT_COILS, DEFAULT_FREQUENCIES,
        )
        assert len(meas) == 27


# ===========================================================================
# TestERTMeasurementGeneration
# ===========================================================================

class TestERTMeasurementGeneration:
    """Validates generate_ert_measurements."""

    def test_correct_count_two_probes(self, two_probes):
        """8 rings/probe → 7 adjacent pairs, 7×7×2 = 98 cross-hole."""
        meas = generate_ert_measurements(two_probes)
        assert len(meas) == 98

    def test_cross_borehole_constraint(self, ert_measurements):
        """Current and potential pairs are from different boreholes."""
        for m in ert_measurements:
            # C1 and C2 share the same borehole letter prefix
            c_bh = m.c1_label[0]
            assert m.c2_label[0] == c_bh
            # P1 and P2 share a different prefix
            p_bh = m.p1_label[0]
            assert m.p2_label[0] == p_bh
            assert c_bh != p_bh

    def test_3d_positions(self, ert_measurements):
        """All electrode positions are 3D arrays."""
        for m in ert_measurements:
            assert m.c1_position.shape == (3,)
            assert m.c2_position.shape == (3,)
            assert m.p1_position.shape == (3,)
            assert m.p2_position.shape == (3,)

    def test_no_duplicate_quadrupoles(self, ert_measurements):
        """No duplicate (c1, c2, p1, p2) index tuples."""
        seen = set()
        for m in ert_measurements:
            key = (m.c1_index, m.c2_index, m.p1_index, m.p2_index)
            assert key not in seen, f"Duplicate quadrupole: {key}"
            seen.add(key)

    def test_crosshole_supports_three_probes(self):
        """Cross-hole generation includes all probes (A/B/C), not only A/B."""
        probes = [
            ProbeConfig(position=[0.0, 0.0, 0.0], length=1.0, ring_depths=[0.1, 0.2, 0.3]),
            ProbeConfig(position=[1.0, 0.0, 0.0], length=1.0, ring_depths=[0.1, 0.2, 0.3]),
            ProbeConfig(position=[2.0, 0.0, 0.0], length=1.0, ring_depths=[0.1, 0.2, 0.3]),
        ]
        meas = generate_ert_measurements(probes, array_type="crosshole")
        letters = {m.c1_label[0] for m in meas} | {m.p1_label[0] for m in meas}
        assert letters == {"A", "B", "C"}

    def test_wenner_generation(self, two_probes):
        """Wenner array_type produces local-borehole measurements."""
        meas = generate_ert_measurements(two_probes, array_type="wenner")
        assert len(meas) > 0
        for m in meas:
            assert m.c1_label[0] == m.c2_label[0] == m.p1_label[0] == m.p2_label[0]

    def test_dipole_dipole_generation(self, two_probes):
        """Dipole-dipole array_type produces local-borehole measurements."""
        meas = generate_ert_measurements(two_probes, array_type="dipole-dipole")
        assert len(meas) > 0
        for m in meas:
            assert m.c1_label[0] == m.c2_label[0] == m.p1_label[0] == m.p2_label[0]


# ===========================================================================
# TestSimulateFDEM
# ===========================================================================

class TestSimulateFDEM:
    """Validates simulate_fdem physics and output structure."""

    def _run_fdem(self, two_probes, cond_model, em_sources=None,
                  add_noise=False, seed=42):
        meas = generate_fdem_measurements(
            two_probes, DEFAULT_COILS, DEFAULT_FREQUENCIES,
        )
        config = HIRTSurveyConfig(frequencies=DEFAULT_FREQUENCIES)
        rng = np.random.default_rng(seed)
        return simulate_fdem(
            meas, cond_model, em_sources or [], [], config, rng, add_noise,
        )

    def test_homogeneous_response(self, two_probes, simple_cond_model):
        """Homogeneous half-space with no spheres: responses are finite."""
        data = self._run_fdem(two_probes, simple_cond_model)
        reals = np.array(data['response_real'])
        assert np.all(np.isfinite(reals))

    def test_sphere_effect(self, two_probes):
        """Sphere near measurement midpoint increases response magnitude."""
        cond_model = {'thicknesses': [], 'conductivities': [0.01]}
        data_no_sphere = self._run_fdem(two_probes, cond_model, [])

        # Place a large conductive sphere between the probes
        sphere = [{
            'position': [12.5, 12.5, -0.5],
            'radius': 0.15,
            'conductivity': 1e6,
        }]
        data_with_sphere = self._run_fdem(two_probes, cond_model, sphere)

        mag_no = np.sqrt(
            np.array(data_no_sphere['response_real'])**2
            + np.array(data_no_sphere['response_imag'])**2
        )
        mag_with = np.sqrt(
            np.array(data_with_sphere['response_real'])**2
            + np.array(data_with_sphere['response_imag'])**2
        )
        # At least some measurements should show increased magnitude
        assert np.any(mag_with > mag_no)

    def test_distance_falloff(self, two_probes):
        """Sphere closer to midpoint → stronger response than far sphere."""
        cond_model = {'thicknesses': [], 'conductivities': [0.01]}
        near = [{'position': [12.5, 12.5, -0.5], 'radius': 0.1,
                 'conductivity': 1e6}]
        far = [{'position': [20.0, 20.0, -0.5], 'radius': 0.1,
                'conductivity': 1e6}]

        data_near = self._run_fdem(two_probes, cond_model, near)
        data_far = self._run_fdem(two_probes, cond_model, far)

        mag_near = np.max(np.sqrt(
            np.array(data_near['response_real'])**2
            + np.array(data_near['response_imag'])**2
        ))
        mag_far = np.max(np.sqrt(
            np.array(data_far['response_real'])**2
            + np.array(data_far['response_imag'])**2
        ))
        assert mag_near > mag_far

    def test_frequency_effect(self, two_probes, simple_cond_model):
        """Different frequencies produce different skin depths."""
        data = self._run_fdem(two_probes, simple_cond_model)
        skin_depths = data['skin_depth']
        unique_sd = set(skin_depths)
        assert len(unique_sd) == len(DEFAULT_FREQUENCIES)

    def test_output_keys_and_shapes(self, two_probes, simple_cond_model):
        """Output dict has expected keys with consistent lengths."""
        data = self._run_fdem(two_probes, simple_cond_model)
        expected_keys = {
            'measurement_id', 'probe_pair', 'tx_depth', 'rx_depth',
            'coil_separation', 'frequency', 'response_real',
            'response_imag', 'skin_depth',
        }
        assert expected_keys == set(data.keys())
        n = len(data['measurement_id'])
        assert n == 108
        for key in expected_keys:
            assert len(data[key]) == n

    def test_no_nan(self, two_probes, simple_cond_model):
        """No NaN values in output."""
        data = self._run_fdem(two_probes, simple_cond_model)
        for key in ('response_real', 'response_imag', 'skin_depth',
                     'tx_depth', 'rx_depth', 'coil_separation'):
            assert not any(np.isnan(v) for v in data[key])

    def test_noise_effects(self, two_probes, simple_cond_model):
        """With noise, results differ from clean."""
        clean = self._run_fdem(
            two_probes, simple_cond_model, add_noise=False,
        )
        noisy = self._run_fdem(
            two_probes, simple_cond_model, add_noise=True, seed=42,
        )
        assert not np.allclose(
            clean['response_real'], noisy['response_real'],
        )

    def test_seed_reproducibility(self, two_probes, simple_cond_model):
        """Same seed → identical noisy results."""
        data1 = self._run_fdem(
            two_probes, simple_cond_model, add_noise=True, seed=123,
        )
        data2 = self._run_fdem(
            two_probes, simple_cond_model, add_noise=True, seed=123,
        )
        np.testing.assert_array_equal(
            data1['response_real'], data2['response_real'],
        )

    def test_coil_depth_overrides_affect_positions(self):
        """Probe-specific coil_depths alter generated FDEM geometry."""
        probes_a = [
            ProbeConfig(position=[0.0, 0.0, 0.0], length=1.0, coil_depths=[0.1, 0.25, 0.4, 0.55, 0.7, 0.85]),
            ProbeConfig(position=[1.0, 0.0, 0.0], length=1.0, coil_depths=[0.1, 0.25, 0.4, 0.55, 0.7, 0.85]),
        ]
        probes_b = [
            ProbeConfig(position=[0.0, 0.0, 0.0], length=1.0, coil_depths=[0.2, 0.35, 0.5, 0.65, 0.8, 0.95]),
            ProbeConfig(position=[1.0, 0.0, 0.0], length=1.0, coil_depths=[0.2, 0.35, 0.5, 0.65, 0.8, 0.95]),
        ]
        m_a = generate_fdem_measurements(probes_a, DEFAULT_COILS, [1000.0])
        m_b = generate_fdem_measurements(probes_b, DEFAULT_COILS, [1000.0])
        assert not np.allclose(m_a[0].tx_position, m_b[0].tx_position)

    def test_anomaly_zone_changes_fdem_response(self, two_probes, simple_cond_model):
        """Conductive anomaly zones perturb FDEM outputs."""
        meas = generate_fdem_measurements(two_probes, DEFAULT_COILS, DEFAULT_FREQUENCIES)
        cfg = HIRTSurveyConfig(frequencies=DEFAULT_FREQUENCIES)
        clean = simulate_fdem(
            meas, simple_cond_model, [], [], cfg, np.random.default_rng(42), False
        )
        zones = [
            AnomalyZone(
                name="zone",
                center=[12.5, 12.5, -0.6],
                dimensions={"radius": 1.0},
                shape="sphere",
                conductivity=0.8,
                resistivity=1.25,
            )
        ]
        pert = simulate_fdem(
            meas, simple_cond_model, [], zones, cfg, np.random.default_rng(42), False
        )
        assert not np.allclose(clean["response_real"], pert["response_real"])


# ===========================================================================
# TestSimulateERT
# ===========================================================================

class TestSimulateERT:
    """Validates simulate_ert physics and output structure."""

    def _run_ert(self, two_probes, res_model, add_noise=False, seed=42):
        meas = generate_ert_measurements(two_probes)
        config = HIRTSurveyConfig()
        rng = np.random.default_rng(seed)
        return simulate_ert(meas, res_model, [], config, rng, add_noise)

    def test_halfspace_returns_true_rho(self, two_probes):
        """Homogeneous half-space: ρ_a ≈ ρ_true."""
        res_model = {'thicknesses': [], 'resistivities': [100.0]}
        data = self._run_ert(two_probes, res_model, add_noise=False)
        rho_a = np.array(data['apparent_resistivity'])
        np.testing.assert_allclose(rho_a, 100.0, rtol=1e-6)

    def test_positive_geometric_factors(self, two_probes):
        """All geometric factors are positive (cross-hole geometry)."""
        res_model = {'thicknesses': [], 'resistivities': [100.0]}
        data = self._run_ert(two_probes, res_model, add_noise=False)
        k_factors = np.array(data['geometric_factor'])
        assert np.all(k_factors > 0)

    def test_output_keys(self, two_probes, simple_resistivity_model):
        """Output dict has expected keys."""
        data = self._run_ert(two_probes, simple_resistivity_model)
        expected = {
            'measurement_id', 'c1_electrode', 'c2_electrode',
            'p1_electrode', 'p2_electrode', 'geometric_factor',
            'apparent_resistivity',
        }
        assert expected == set(data.keys())

    def test_no_nan(self, two_probes, simple_resistivity_model):
        """No NaN in apparent resistivity or geometric factors."""
        data = self._run_ert(two_probes, simple_resistivity_model)
        assert not any(np.isnan(v) for v in data['apparent_resistivity'])
        assert not any(np.isnan(v) for v in data['geometric_factor'])

    def test_noise_changes_results(self, two_probes, simple_resistivity_model):
        """With noise, apparent resistivities deviate from clean."""
        clean = self._run_ert(
            two_probes, simple_resistivity_model, add_noise=False,
        )
        noisy = self._run_ert(
            two_probes, simple_resistivity_model, add_noise=True, seed=42,
        )
        assert not np.allclose(
            clean['apparent_resistivity'], noisy['apparent_resistivity'],
        )

    def test_electrode_labels(self, two_probes, simple_resistivity_model):
        """Electrode labels follow A/B prefix convention."""
        data = self._run_ert(two_probes, simple_resistivity_model)
        for label in data['c1_electrode'] + data['p1_electrode']:
            assert label[0] in ('A', 'B')

    def test_anomaly_zone_changes_ert_response(self, two_probes, simple_resistivity_model):
        """Anomaly zones perturb apparent resistivity."""
        meas = generate_ert_measurements(two_probes)
        cfg = HIRTSurveyConfig()
        clean = simulate_ert(
            meas, simple_resistivity_model, [], cfg, np.random.default_rng(7), False
        )
        zones = [
            AnomalyZone(
                name="wet_fill",
                center=[12.5, 12.5, -0.5],
                dimensions={"length": 2.0, "width": 2.0, "depth": 1.0},
                shape="box",
                conductivity=1.0,
                resistivity=1.0,
            )
        ]
        pert = simulate_ert(
            meas, simple_resistivity_model, zones, cfg, np.random.default_rng(7), False
        )
        assert not np.allclose(
            clean["apparent_resistivity"], pert["apparent_resistivity"]
        )


# ===========================================================================
# TestExportCSV
# ===========================================================================

class TestExportCSV:
    """Validates FDEM and ERT CSV export."""

    def _make_fdem_data(self, n=5):
        return {
            'measurement_id': list(range(n)),
            'probe_pair': ['0-0'] * n,
            'tx_depth': [0.1 * i for i in range(n)],
            'rx_depth': [0.2 * i for i in range(n)],
            'coil_separation': [0.15] * n,
            'frequency': [1000.0] * n,
            'response_real': [1e-3 * i for i in range(n)],
            'response_imag': [2e-3 * i for i in range(n)],
            'skin_depth': [5.0] * n,
        }

    def _make_ert_data(self, n=5):
        return {
            'measurement_id': list(range(n)),
            'c1_electrode': ['A1'] * n,
            'c2_electrode': ['A2'] * n,
            'p1_electrode': ['B1'] * n,
            'p2_electrode': ['B2'] * n,
            'geometric_factor': [10.0] * n,
            'apparent_resistivity': [100.0] * n,
        }

    def test_fdem_csv_created(self, tmp_path):
        """FDEM CSV file is created and non-empty."""
        path = tmp_path / "fdem.csv"
        export_fdem_csv(self._make_fdem_data(), path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_ert_csv_created(self, tmp_path):
        """ERT CSV file is created and non-empty."""
        path = tmp_path / "ert.csv"
        export_ert_csv(self._make_ert_data(), path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_correct_headers(self, tmp_path):
        """CSV files have expected column headers."""
        fdem_path = tmp_path / "fdem.csv"
        ert_path = tmp_path / "ert.csv"
        export_fdem_csv(self._make_fdem_data(), fdem_path)
        export_ert_csv(self._make_ert_data(), ert_path)

        with open(fdem_path) as f:
            header = f.readline().strip()
        assert header == (
            "measurement_id,probe_pair,tx_depth,rx_depth,"
            "coil_separation,frequency,response_real,response_imag,skin_depth"
        )

        with open(ert_path) as f:
            header = f.readline().strip()
        assert header == (
            "measurement_id,c1_electrode,c2_electrode,"
            "p1_electrode,p2_electrode,geometric_factor,apparent_resistivity"
        )

    def test_pandas_loadable(self, tmp_path):
        """Both CSVs are loadable by pandas with correct row counts."""
        pd = pytest.importorskip("pandas")
        n = 10

        fdem_path = tmp_path / "fdem.csv"
        export_fdem_csv(self._make_fdem_data(n), fdem_path)
        df_fdem = pd.read_csv(fdem_path)
        assert len(df_fdem) == n

        ert_path = tmp_path / "ert.csv"
        export_ert_csv(self._make_ert_data(n), ert_path)
        df_ert = pd.read_csv(ert_path)
        assert len(df_ert) == n

    def test_hirt_record_exports(self, tmp_path):
        """HIRT docs-compatible MIT/ERT exports are created."""
        mit_path = tmp_path / "mit_records.csv"
        ert_path = tmp_path / "ert_records.csv"
        export_hirt_mit_csv(self._make_fdem_data(), mit_path)
        export_hirt_ert_csv(self._make_ert_data(), ert_path)
        assert mit_path.exists()
        assert ert_path.exists()
        with open(mit_path) as f:
            assert f.readline().strip().startswith("timestamp,section_id,zone_id")
        with open(ert_path) as f:
            assert f.readline().strip().startswith("timestamp,section_id,zone_id")

    def test_ert_record_export_includes_polarity_reversal_rows(self, tmp_path):
        """ERT record export writes + and - polarity rows by default."""
        ert_path = tmp_path / "ert_records.csv"
        export_hirt_ert_csv(self._make_ert_data(n=3), ert_path)
        with open(ert_path) as f:
            rows = f.readlines()
        # header + 2 rows per measurement
        assert len(rows) == 1 + 3 * 2


# ===========================================================================
# TestRunHIRTSurvey
# ===========================================================================

class TestRunHIRTSurvey:
    """Validates run_hirt_survey end-to-end."""

    def test_both_csvs_created(self, tmp_path):
        """Both FDEM and ERT CSVs are created."""
        result = run_hirt_survey(SWAMP_SCENARIO, tmp_path, seed=42)
        assert Path(result['fdem_csv']).exists()
        assert Path(result['ert_csv']).exists()

    def test_fdem_only_mode(self, tmp_path):
        """ert_enabled=False produces only FDEM output."""
        result = run_hirt_survey(
            SWAMP_SCENARIO, tmp_path, ert_enabled=False, seed=42,
        )
        assert 'fdem' in result
        assert 'ert' not in result

    def test_ert_only_mode(self, tmp_path):
        """fdem_enabled=False produces only ERT output."""
        result = run_hirt_survey(
            SWAMP_SCENARIO, tmp_path, fdem_enabled=False, seed=42,
        )
        assert 'ert' in result
        assert 'fdem' not in result

    def test_raises_without_hirt_config(self, tmp_path):
        """Scenario without hirt_config raises ValueError."""
        import json
        # Create a minimal scenario with no hirt_config
        no_hirt = {
            "name": "No HIRT",
            "description": "Test scenario without hirt_config",
            "earth_field": [0.0, 20e-6, 45e-6],
            "terrain": {
                "x_extent": [0, 10], "y_extent": [0, 10],
                "surface_elevation": 0.0,
                "layers": [{"name": "Soil", "z_top": 0, "z_bottom": -2,
                            "conductivity": 0.05}],
            },
            "objects": [{"name": "Sphere", "position": [5, 5, -1],
                         "type": "ferrous_sphere", "radius": 0.05,
                         "susceptibility": 1000, "conductivity": 1e6}],
        }
        no_hirt_path = tmp_path / "no_hirt.json"
        no_hirt_path.write_text(json.dumps(no_hirt))
        with pytest.raises(ValueError, match="hirt_config"):
            run_hirt_survey(no_hirt_path, tmp_path / "out", seed=42)

    def test_seed_reproducibility(self, tmp_path):
        """Same seed → identical results."""
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        result1 = run_hirt_survey(SWAMP_SCENARIO, out1, seed=99)
        result2 = run_hirt_survey(SWAMP_SCENARIO, out2, seed=99)
        np.testing.assert_array_equal(
            result1['fdem']['response_real'],
            result2['fdem']['response_real'],
        )
        np.testing.assert_array_equal(
            result1['ert']['apparent_resistivity'],
            result2['ert']['apparent_resistivity'],
        )

    def test_return_dict_structure(self, tmp_path):
        """Return dict has expected keys and non-empty data."""
        result = run_hirt_survey(SWAMP_SCENARIO, tmp_path, seed=42)
        assert 'fdem' in result
        assert 'ert' in result
        assert 'fdem_csv' in result
        assert 'ert_csv' in result
        assert 'mit_records_csv' in result
        assert 'ert_records_csv' in result
        assert len(result['fdem']['measurement_id']) > 0
        assert len(result['ert']['measurement_id']) > 0

    def test_sidecar_exports_present(self, tmp_path):
        """Survey sidecar exports (geometry, registry, QC) are produced."""
        result = run_hirt_survey(SWAMP_SCENARIO, tmp_path, seed=42)
        assert 'survey_geometry_csv' in result
        assert 'probe_registry_csv' in result
        assert 'qc_summary_json' in result
        assert Path(result['survey_geometry_csv']).exists()
        assert Path(result['probe_registry_csv']).exists()
        assert Path(result['qc_summary_json']).exists()
