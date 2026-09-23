"""Tests for the benchtop-rig telemetry layer.

These are deliberately adversarial: the protocol must reject bad frames rather
than accept them, the calibration must refuse to extrapolate silently, the
pulse detector must recover a known heart rate, and the validation report must
show a *non-zero* bias when the rig is given a reality gap. A telemetry layer
that only works on clean synthetic input is worse than none, because it makes
a broken rig look like a working one.
"""
import math

import numpy as np
import pytest

from danon.telemetry import (
    ARDUINO_FIRMWARE,
    CHANNEL_UNITS,
    DEFAULT_CHANNELS,
    SENSOR_SPECS,
    WIRING_TABLE,
    AcquisitionSession,
    LineProtocol,
    ProtocolError,
    ReplayBackend,
    RigConfig,
    RigTrace,
    SensorCalibration,
    SimulatedRigBackend,
    clearance_for_reduction,
    default_rig_config,
    detect_pulse_peaks,
    nominal_reduction_pct,
    run_validation,
    validate_against_prediction,
    waveform_stats,
    xor_checksum,
)


# ──────────────────────────────────────────────────────────────────────────────
# Protocol
# ──────────────────────────────────────────────────────────────────────────────


def _counts(offset: float = 0.0) -> dict:
    return {
        "p_proximal": 12000.0 + offset,
        "p_distal": 9000.0 + offset,
        "temperature": 592.0,
        "flow": 110.0,
    }


class TestLineProtocol:
    def test_round_trip_preserves_values(self):
        proto = LineProtocol()
        line = proto.encode(42, 1.234, _counts())
        sample = proto.decode(line)
        assert sample.seq == 42
        assert sample.t_s == pytest.approx(1.234)
        for name in DEFAULT_CHANNELS:
            assert sample.raw_counts[name] == pytest.approx(_counts()[name])

    def test_wire_format_matches_firmware_field_order(self):
        proto = LineProtocol()
        line = proto.encode(1, 0.0, _counts())
        body = line[1:].split("*")[0]
        fields = body.split(",")
        # seq, t_ms, then one field per physical channel
        assert len(fields) == 2 + len(DEFAULT_CHANNELS)
        assert len(DEFAULT_CHANNELS) == 4

    def test_checksum_is_xor_of_body(self):
        proto = LineProtocol()
        line = proto.encode(3, 0.5, _counts())
        body = line[1:].split("*")[0]
        assert f"{xor_checksum(body):02X}" in line

    def test_corrupted_checksum_is_rejected(self):
        proto = LineProtocol()
        line = proto.encode(7, 0.1, _counts())
        corrupted = line[:-2] + ("00" if not line.endswith("00") else "FF")
        with pytest.raises(ProtocolError, match="checksum"):
            proto.decode(corrupted)

    def test_corrupted_payload_is_rejected_not_silently_accepted(self):
        proto = LineProtocol()
        line = proto.encode(7, 0.1, _counts())
        # flip a digit inside the payload, leaving the old checksum
        idx = line.index("12") + 1
        mutated = line[:idx] + ("9" if line[idx] != "9" else "8") + line[idx + 1:]
        with pytest.raises(ProtocolError, match="checksum"):
            proto.decode(mutated)

    @pytest.mark.parametrize("bad", ["", "   ", "garbage", "#1,2,3,4,5,6"])
    def test_malformed_lines_raise(self, bad):
        with pytest.raises(ProtocolError):
            LineProtocol().decode(bad)

    def test_non_hex_checksum_raises(self):
        with pytest.raises(ProtocolError, match="non-hex"):
            LineProtocol().decode("#1,2,3,4,5,6*ZZ")

    def test_wrong_field_count_raises(self):
        body = "1,2,3,4,5"
        with pytest.raises(ProtocolError, match="fields"):
            LineProtocol().decode(f"#{body}*{xor_checksum(body):02X}")

    def test_non_numeric_field_raises(self):
        body = "1,2,abc,4,5,6"
        with pytest.raises(ProtocolError, match="non-numeric"):
            LineProtocol().decode(f"#{body}*{xor_checksum(body):02X}")


# ──────────────────────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────────────────────


class TestCalibration:
    def test_span_calibration_recovers_known_pressure(self):
        cal = SensorCalibration.from_span(
            counts_at_zero=3277.0, counts_at_full_scale=26214.0, full_scale_value=40.0)
        assert cal.apply(3277.0) == pytest.approx(0.0, abs=1e-9)
        assert cal.apply(26214.0) == pytest.approx(40.0, abs=1e-9)
        assert cal.apply((3277.0 + 26214.0) / 2) == pytest.approx(20.0, abs=1e-9)

    def test_span_calibration_rejects_degenerate_points(self):
        with pytest.raises(ValueError, match="differ"):
            SensorCalibration.from_span(
                counts_at_zero=100.0, counts_at_full_scale=100.0, full_scale_value=40.0)

    def test_least_squares_fit_recovers_slope_and_reports_quality(self):
        rng = np.random.default_rng(0)
        truth = np.linspace(0.0, 40.0, 12)
        slope_true, intercept_true = 0.0017, -5.6
        counts = (truth - intercept_true) / slope_true
        measured = truth + rng.normal(0.0, 0.03, truth.size)

        fit = SensorCalibration.fit(counts, measured)
        assert fit.slope == pytest.approx(slope_true, rel=1e-3)
        assert fit.intercept == pytest.approx(intercept_true, abs=0.5)
        assert fit.r_squared > 0.99
        assert fit.residual_std < 0.1

    def test_fit_requires_matching_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            SensorCalibration.fit([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_fit_rejects_nan(self):
        with pytest.raises(ValueError, match="non-finite"):
            SensorCalibration.fit([1.0, float("nan")], [0.0, 1.0])

    def test_resolution_is_one_count_in_engineering_units(self):
        cal = SensorCalibration.from_span(
            counts_at_zero=0.0, counts_at_full_scale=1000.0, full_scale_value=10.0)
        assert cal.resolution == pytest.approx(0.01)

    def test_extrapolation_is_flagged(self, caplog):
        cal = SensorCalibration.from_span(
            counts_at_zero=1000.0, counts_at_full_scale=2000.0, full_scale_value=10.0)
        with caplog.at_level("WARNING"):
            cal.apply(5000.0)
        assert any("extrapolation" in r.message for r in caplog.records)


class TestRigConfig:
    def test_default_config_covers_every_channel(self):
        cfg = default_rig_config()
        for name in DEFAULT_CHANNELS:
            assert cfg.channel(name).unit == CHANNEL_UNITS[name]

    def test_unknown_channel_raises(self):
        with pytest.raises(KeyError):
            default_rig_config().channel("nope")

    def test_combined_dp_uncertainty_is_quadrature_sum(self):
        cfg = RigConfig(channels=[
            # two transducers with known residual scatter
            __import__("danon.telemetry", fromlist=["RigChannel"]).RigChannel(
                "p_proximal", "kPa",
                SensorCalibration(slope=1.0, intercept=0.0, residual_std=0.03)),
            __import__("danon.telemetry", fromlist=["RigChannel"]).RigChannel(
                "p_distal", "kPa",
                SensorCalibration(slope=1.0, intercept=0.0, residual_std=0.04)),
        ])
        assert cfg.nominal_dp_uncertainty_kpa == pytest.approx(math.hypot(0.03, 0.04))


# ──────────────────────────────────────────────────────────────────────────────
# Waveform analysis
# ──────────────────────────────────────────────────────────────────────────────


class TestPulseDetection:
    @pytest.mark.parametrize("bpm", [50.0, 70.0, 120.0, 180.0])
    def test_recovers_known_heart_rate(self, bpm):
        fs = 200.0
        t = np.arange(0.0, 10.0, 1.0 / fs)
        signal = 1.0 + 0.3 * np.sin(2 * math.pi * bpm / 60.0 * t)
        peaks = detect_pulse_peaks(signal, fs)
        assert peaks.size >= 5
        est = 60.0 / float(np.median(np.diff(peaks) / fs))
        assert est == pytest.approx(bpm, rel=0.02)

    def test_flat_signal_yields_no_beats(self):
        assert detect_pulse_peaks(np.ones(500), 200.0).size <= 1

    def test_short_signal_is_handled(self):
        assert detect_pulse_peaks(np.array([1.0]), 200.0).size == 0

    def test_waveform_stats_are_ordered(self):
        fs = 200.0
        t = np.arange(0.0, 8.0, 1.0 / fs)
        signal = 10.0 + 2.0 * np.sin(2 * math.pi * 70.0 / 60.0 * t)
        stats = waveform_stats(signal, fs)
        assert stats.diastolic < stats.mean < stats.systolic
        assert stats.pulse_pressure > 0
        assert stats.heart_rate_bpm == pytest.approx(70.0, rel=0.05)
        assert stats.n_beats >= 5

    def test_empty_signal_does_not_crash(self):
        stats = waveform_stats([], 200.0)
        assert stats.n_beats == 0 and stats.mean == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Acquisition
# ──────────────────────────────────────────────────────────────────────────────


class TestAcquisition:
    def test_sample_count_matches_duration_and_rate(self):
        cfg = default_rig_config(sample_rate_hz=200.0)
        trace = AcquisitionSession(cfg, SimulatedRigBackend(cfg, duration_s=2.0)).acquire(2.0)
        assert trace.t_s.size == 400
        assert trace.checksum_errors == 0

    def test_acquisition_recovers_heart_rate_from_hardware_path(self):
        cfg = default_rig_config()
        backend = SimulatedRigBackend(cfg, duration_s=6.0, heart_rate_bpm=90.0)
        metrics = AcquisitionSession(cfg, backend).acquire(6.0).metrics()
        assert metrics.proximal.heart_rate_bpm == pytest.approx(90.0, rel=0.05)

    def test_distal_pressure_is_lower_than_proximal(self):
        cfg = default_rig_config()
        trace = AcquisitionSession(cfg, SimulatedRigBackend(cfg, duration_s=2.0)).acquire(2.0)
        assert np.mean(trace.dp_kpa) > 0

    def test_synthetic_traces_are_labelled_synthetic(self):
        cfg = default_rig_config()
        trace = AcquisitionSession(cfg, SimulatedRigBackend(cfg, duration_s=1.0)).acquire(1.0)
        assert trace.is_synthetic
        assert trace.metrics().is_synthetic

    def test_replay_backend_round_trips_a_capture(self):
        cfg = default_rig_config()
        original = AcquisitionSession(cfg, SimulatedRigBackend(cfg, duration_s=1.0)).acquire(1.0)

        # Rebuild the raw-count log that the rig would have written, then replay it.
        proto = LineProtocol()
        lines = []
        for i in range(original.t_s.size):
            counts = {
                "p_proximal": (original.channel("p_proximal")[i] - cfg.channel("p_proximal").calibration.intercept)
                / cfg.channel("p_proximal").calibration.slope,
                "p_distal": (original.channel("p_distal")[i] - cfg.channel("p_distal").calibration.intercept)
                / cfg.channel("p_distal").calibration.slope,
                "temperature": 592.0,
                "flow": 110.0,
            }
            lines.append(proto.encode(i, float(original.t_s[i]), counts))

        replayed = AcquisitionSession(cfg, ReplayBackend("\n".join(lines))).acquire(100.0)
        assert replayed.t_s.size == original.t_s.size
        assert replayed.channel("p_proximal") == pytest.approx(
            original.channel("p_proximal"), abs=1e-6)

    def test_trace_csv_round_trip(self):
        cfg = default_rig_config()
        trace = AcquisitionSession(cfg, SimulatedRigBackend(cfg, duration_s=1.0)).acquire(1.0)
        restored = RigTrace.from_csv(trace.to_csv(), cfg)
        assert restored.t_s == pytest.approx(trace.t_s)
        for name in DEFAULT_CHANNELS:
            assert restored.channel(name) == pytest.approx(trace.channel(name))


# ──────────────────────────────────────────────────────────────────────────────
# Lumen law
# ──────────────────────────────────────────────────────────────────────────────


class TestLumenLaw:
    def test_zero_clearance_is_zero_reduction(self):
        assert nominal_reduction_pct(0.0) == pytest.approx(0.0)

    def test_reduction_increases_monotonically_with_clearance(self):
        values = [nominal_reduction_pct(c) for c in range(0, 71, 5)]
        assert all(b > a for a, b in zip(values, values[1:]))

    def test_clearance_inversion_round_trips(self):
        for c in (5.0, 20.0, 45.0, 70.0):
            assert clearance_for_reduction(nominal_reduction_pct(c)) == pytest.approx(c, abs=1e-6)

    def test_restoring_ten_percent_of_the_lumen_buys_more_than_ten_percent(self):
        # the fourth-power law is the headline physiological result
        assert nominal_reduction_pct(10.0) > 30.0

    def test_unreachable_reduction_is_rejected(self):
        with pytest.raises(ValueError):
            clearance_for_reduction(100.0)

    def test_pressure_drop_follows_inverse_fourth_power(self):
        cfg = default_rig_config()
        dp = {}
        for clear in (0.0, 30.0):
            backend = SimulatedRigBackend(
                cfg, plaque_clear_pct=clear, duration_s=2.0, viscosity_error=0.0,
                run_offset_kpa=0.0, zero_drift_kpa=0.0, noise_kpa=0.0, seed=0)
            dp[clear] = AcquisitionSession(cfg, backend).acquire(2.0).metrics().mean_dp_kpa

        r0 = math.sqrt(1.0 - 0.70)
        area = 0.70 * (1.0 - 0.30)
        r1 = math.sqrt(1.0 - area)
        assert dp[0.0] / dp[30.0] == pytest.approx((r1 / r0) ** 4, rel=1e-3)


# ──────────────────────────────────────────────────────────────────────────────
# Validation statistics
# ──────────────────────────────────────────────────────────────────────────────


class TestValidation:
    def test_perfect_agreement_gives_r2_near_one_and_zero_bias(self):
        values = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        report = validate_against_prediction(values, values, n_bootstrap=200)
        assert report.r_squared == pytest.approx(1.0, abs=1e-6)
        assert report.bias == pytest.approx(0.0, abs=1e-9)
        assert report.slope == pytest.approx(1.0, abs=1e-6)

    def test_constant_offset_shows_up_as_bias_not_as_bad_correlation(self):
        predicted = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        measured = [p + 5.0 for p in predicted]
        report = validate_against_prediction(predicted, measured, n_bootstrap=200)
        assert report.r_squared == pytest.approx(1.0, abs=1e-6)
        assert report.bias == pytest.approx(5.0)
        assert report.loa_low == pytest.approx(5.0)
        assert report.loa_high == pytest.approx(5.0)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            validate_against_prediction([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_too_few_points_raise(self):
        with pytest.raises(ValueError, match="three paired"):
            validate_against_prediction([1.0, 2.0], [1.0, 2.0])

    def test_reality_gap_produces_a_slope_below_one(self):
        """A rig that delivers less clearance than commanded must not look perfect."""
        cfg = default_rig_config()
        predictions = [nominal_reduction_pct(c) for c in (0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0)]
        report = run_validation(
            predictions, config=cfg, duration_s=1.5,
            clearance_efficiency=0.90, run_offset_kpa=0.03, seed=1)
        assert report.slope < 1.0
        assert report.bias < 0.0
        assert report.r2_ci_low <= report.r_squared <= report.r2_ci_high

    def test_no_reality_gap_reproduces_the_model(self):
        cfg = default_rig_config()
        predictions = [nominal_reduction_pct(c) for c in (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)]
        report = run_validation(
            predictions, config=cfg, duration_s=1.5,
            clearance_efficiency=1.0, run_offset_kpa=0.0,
            viscosity_error=0.0, seed=2)
        assert report.r_squared > 0.999
        assert report.bias == pytest.approx(0.0, abs=0.5)

    def test_validation_labels_synthetic_pairs(self):
        cfg = default_rig_config()
        predictions = [nominal_reduction_pct(c) for c in (0.0, 15.0, 30.0, 45.0)]
        report = run_validation(predictions, config=cfg, duration_s=1.0)
        assert report.n_synthetic_pairs == report.n + 1  # one baseline run + one per point

    def test_misaligned_commands_raise(self):
        cfg = default_rig_config()
        with pytest.raises(ValueError, match="align"):
            run_validation([1.0, 2.0, 3.0], config=cfg,
                           commanded_clearance_pct=[1.0, 2.0])


# ──────────────────────────────────────────────────────────────────────────────
# Hardware definition consistency
# ──────────────────────────────────────────────────────────────────────────────


class TestHardwareDefinition:
    def test_firmware_emits_every_protocol_channel(self):
        for name in DEFAULT_CHANNELS:
            # each channel must appear in the field list the firmware prints
            assert name in CHANNEL_UNITS
        assert "snprintf" in ARDUINO_FIRMWARE
        # six numeric fields: seq, t_ms, p1, p2, temp, flow
        assert ARDUINO_FIRMWARE.count("%d") + ARDUINO_FIRMWARE.count("%lu") == 6

    def test_firmware_implements_the_checksum_the_parser_expects(self):
        assert "x ^= (uint8_t)*c" in ARDUINO_FIRMWARE

    def test_sensor_specs_cover_all_channels(self):
        measured = set()
        for spec in SENSOR_SPECS:
            measured.update(str(spec["measures"]).replace(" ", "").split(","))
        assert set(DEFAULT_CHANNELS) <= measured

    def test_wiring_table_is_populated(self):
        assert WIRING_TABLE
        assert all(isinstance(v, str) and v for v in WIRING_TABLE.values())
