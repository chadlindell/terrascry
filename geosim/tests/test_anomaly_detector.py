"""Tests for real-time anomaly detection.

Validates the SimpleAnomalyDetector against known signal patterns:
constant background, single spikes, sustained anomalies, and edge cases.
"""

import pytest

from geosim.streaming.anomaly_detector import AnomalyDetectorConfig, SimpleAnomalyDetector


class TestBackgroundEstimation:
    """Tests for rolling background mean and standard deviation."""

    def test_constant_signal_mean(self):
        """Constant input converges to that value as background mean.

        Uses a value within the initial detection threshold (3 * 0.5 = 1.5 nT
        from the initial mean of 0) so that samples enter the buffer.
        """
        detector = SimpleAnomalyDetector()
        for _ in range(60):
            detector.process_sample(0.5)
        assert detector.background_mean == pytest.approx(0.5, rel=1e-10)

    def test_constant_signal_std(self):
        """Constant input gives std equal to noise floor."""
        detector = SimpleAnomalyDetector()
        for _ in range(60):
            detector.process_sample(0.5)
        assert detector.background_std == pytest.approx(
            detector.config.noise_floor_nt, rel=1e-10
        )

    def test_initial_mean_is_zero(self):
        """Empty buffer gives zero background mean."""
        detector = SimpleAnomalyDetector()
        assert detector.background_mean == 0.0

    def test_initial_std_is_noise_floor(self):
        """With fewer than 3 samples, std is the noise floor."""
        detector = SimpleAnomalyDetector()
        assert detector.background_std == detector.config.noise_floor_nt
        detector.process_sample(1.0)
        assert detector.background_std == detector.config.noise_floor_nt
        detector.process_sample(2.0)
        assert detector.background_std == detector.config.noise_floor_nt

    def test_buffer_respects_window_size(self):
        """Buffer does not grow beyond background_window."""
        config = AnomalyDetectorConfig(background_window=10)
        detector = SimpleAnomalyDetector(config)
        for i in range(50):
            detector.process_sample(float(i))
        assert len(detector.buffer) == 10

    def test_mean_tracks_recent_values(self):
        """After window fills, mean reflects only recent samples.

        Gradually ramps the signal so each step stays within the detection
        threshold and enters the buffer.
        """
        config = AnomalyDetectorConfig(background_window=10)
        detector = SimpleAnomalyDetector(config)
        # Fill with zeros
        for _ in range(20):
            detector.process_sample(0.0)
        # Gradually ramp to 0.5 (within threshold of the drifting mean)
        for _ in range(20):
            detector.process_sample(0.5)
        # Background should now track toward 0.5
        assert detector.background_mean == pytest.approx(0.5, rel=1e-10)


class TestSpikeDetection:
    """Tests for single and sustained spike detection."""

    def test_single_spike_not_confirmed(self):
        """A single spike sample is NOT confirmed (min_anomaly_duration=3)."""
        detector = SimpleAnomalyDetector()
        # Build up background
        for _ in range(60):
            detector.process_sample(0.0)
        # Single large spike
        result = detector.process_sample(1000.0)
        assert result["is_anomaly"] is False

    def test_sustained_spike_is_detected(self):
        """Three consecutive spikes trigger confirmed detection."""
        config = AnomalyDetectorConfig(min_anomaly_duration=3)
        detector = SimpleAnomalyDetector(config)
        # Build up background
        for _ in range(60):
            detector.process_sample(0.0)

        # Three large spikes in a row
        results = []
        for _ in range(3):
            results.append(detector.process_sample(1000.0))

        # First two are not confirmed, third is
        assert results[0]["is_anomaly"] is False
        assert results[1]["is_anomaly"] is False
        assert results[2]["is_anomaly"] is True

    def test_negative_spike_detected(self):
        """Large negative deviations are also detected."""
        detector = SimpleAnomalyDetector()
        for _ in range(60):
            detector.process_sample(100.0)

        results = []
        for _ in range(3):
            results.append(detector.process_sample(-900.0))
        assert results[2]["is_anomaly"] is True

    def test_moderate_signal_not_flagged(self):
        """Signal within threshold is not flagged."""
        config = AnomalyDetectorConfig(detection_threshold_sigma=3.0, noise_floor_nt=1.0)
        detector = SimpleAnomalyDetector(config)
        for _ in range(60):
            detector.process_sample(0.0)
        # Small deviation, well within 3 sigma of noise floor
        result = detector.process_sample(1.0)
        assert result["is_anomaly"] is False


class TestMinAnomalyDuration:
    """Tests for min_anomaly_duration filtering."""

    def test_duration_1_confirms_immediately(self):
        """With min_anomaly_duration=1, first spike is confirmed."""
        config = AnomalyDetectorConfig(min_anomaly_duration=1)
        detector = SimpleAnomalyDetector(config)
        for _ in range(60):
            detector.process_sample(0.0)
        result = detector.process_sample(1000.0)
        assert result["is_anomaly"] is True

    def test_duration_5_requires_five_consecutive(self):
        """With min_anomaly_duration=5, need five in a row."""
        config = AnomalyDetectorConfig(min_anomaly_duration=5)
        detector = SimpleAnomalyDetector(config)
        for _ in range(60):
            detector.process_sample(0.0)

        results = [detector.process_sample(1000.0) for _ in range(5)]
        for i in range(4):
            assert results[i]["is_anomaly"] is False
        assert results[4]["is_anomaly"] is True

    def test_interrupted_sequence_resets(self):
        """A non-anomaly sample resets the consecutive counter."""
        config = AnomalyDetectorConfig(min_anomaly_duration=3)
        detector = SimpleAnomalyDetector(config)
        for _ in range(60):
            detector.process_sample(0.0)

        # Two spikes, then normal, then two more spikes
        detector.process_sample(1000.0)
        detector.process_sample(1000.0)
        detector.process_sample(0.0)  # Resets counter
        r1 = detector.process_sample(1000.0)
        r2 = detector.process_sample(1000.0)

        # Neither should be confirmed because counter reset
        assert r1["is_anomaly"] is False
        assert r2["is_anomaly"] is False


class TestBackgroundUpdateDuringAnomaly:
    """Tests that background is not corrupted by anomaly samples."""

    def test_background_unchanged_during_anomaly(self):
        """Background mean does not change during anomaly samples."""
        detector = SimpleAnomalyDetector()
        for _ in range(60):
            detector.process_sample(0.0)
        bg_before = detector.background_mean

        # Feed anomaly samples
        for _ in range(10):
            detector.process_sample(5000.0)

        bg_after = detector.background_mean
        assert bg_after == pytest.approx(bg_before, rel=1e-10)

    def test_buffer_length_unchanged_during_anomaly(self):
        """Buffer length does not grow during anomaly."""
        config = AnomalyDetectorConfig(background_window=50)
        detector = SimpleAnomalyDetector(config)
        for _ in range(60):
            detector.process_sample(0.0)
        buffer_len_before = len(detector.buffer)

        for _ in range(20):
            detector.process_sample(5000.0)

        assert len(detector.buffer) == buffer_len_before


class TestReset:
    """Tests for detector state reset."""

    def test_reset_clears_buffer(self):
        """After reset, buffer is empty."""
        detector = SimpleAnomalyDetector()
        # Use values near zero so they enter the buffer (within threshold)
        for _ in range(30):
            detector.process_sample(0.1)
        assert len(detector.buffer) > 0

        detector.reset()
        assert len(detector.buffer) == 0

    def test_reset_clears_anomaly_count(self):
        """After reset, anomaly_count is zero."""
        detector = SimpleAnomalyDetector()
        for _ in range(60):
            detector.process_sample(0.0)
        for _ in range(5):
            detector.process_sample(5000.0)
        assert detector.anomaly_count > 0

        detector.reset()
        assert detector.anomaly_count == 0

    def test_reset_clears_consecutive_counter(self):
        """After reset, consecutive_anomaly is zero."""
        detector = SimpleAnomalyDetector()
        for _ in range(60):
            detector.process_sample(0.0)
        detector.process_sample(5000.0)
        assert detector.consecutive_anomaly > 0

        detector.reset()
        assert detector.consecutive_anomaly == 0

    def test_reset_returns_to_initial_state(self):
        """After reset, background_mean is zero and std is noise floor."""
        detector = SimpleAnomalyDetector()
        for _ in range(60):
            detector.process_sample(42.0)
        detector.reset()
        assert detector.background_mean == 0.0
        assert detector.background_std == detector.config.noise_floor_nt


class TestResultDict:
    """Tests for the structure of process_sample return value."""

    def test_result_keys(self):
        """Result contains all expected keys."""
        detector = SimpleAnomalyDetector()
        result = detector.process_sample(1.0)
        expected_keys = {
            "is_anomaly",
            "gradient_nt",
            "background_nt",
            "residual_nt",
            "sigma",
            "threshold_nt",
        }
        assert set(result.keys()) == expected_keys

    def test_gradient_nt_echoed(self):
        """Result echoes back the input gradient value."""
        detector = SimpleAnomalyDetector()
        result = detector.process_sample(123.456)
        assert result["gradient_nt"] == 123.456

    def test_sigma_is_positive(self):
        """Sigma is non-negative."""
        detector = SimpleAnomalyDetector()
        for _ in range(60):
            detector.process_sample(0.0)
        result = detector.process_sample(10.0)
        assert result["sigma"] >= 0.0

    def test_threshold_is_positive(self):
        """Threshold is always positive."""
        detector = SimpleAnomalyDetector()
        result = detector.process_sample(0.0)
        assert result["threshold_nt"] > 0.0
