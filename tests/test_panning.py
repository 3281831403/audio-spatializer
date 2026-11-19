"""Contract tests for ``spatial.panning`` -- 15 tests, all initially red.

Coordinates are metres in a listener-aligned right-handed frame (``+X`` right,
``+Y`` up, ``+Z`` behind), velocities are m/s, angles are degrees and times are
seconds.  There is no randomness, no sleeping, no network access and no
external dependency beyond pytest.

The implementation in ``spatial/panning.py`` is a set of
``NotImplementedError`` stubs, so the expected first run is exactly
``15 failed`` with no collection, import or syntax errors.  This file is the
acceptance spec and must not be modified while fixing the spatializer.
"""

from __future__ import annotations

import math

import pytest

from spatial.panning import (
    SpatialError,
    azimuth_elevation,
    distance_gain,
    doppler_ratio,
    hrtf_select,
    itd_delay,
    occlusion_lowpass,
    pan_gains,
    spatialize_batch,
)


LISTENER = {"pos": [0.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0]}

HRTF_BANK = [
    {"name": "front", "azimuth": 0.0, "elevation": 0.0},
    {"name": "left", "azimuth": -90.0, "elevation": 0.0},
    {"name": "right", "azimuth": 90.0, "elevation": 0.0},
]

CONFIG = {
    "layout": "5.1",
    "model": "inverse",
    "ref_distance": 1.0,
    "rolloff": 1.0,
    "sound_speed": 343.0,
    "head_radius": 0.0875,
    "max_occlusion_distance": 10.0,
    "cutoff_hz": 8000.0,
    "hrtf_bank": HRTF_BANK,
}


# ---------------------------------------------------------------------------
# One test per documented defect (10)
# ---------------------------------------------------------------------------


def test_distance_gain_clamps_zero_distance():
    """Defect 1: a zero distance must never divide by zero or return inf."""
    gain = distance_gain(0.0, "inverse", 1.0, 1.0)

    assert math.isfinite(gain)
    assert gain == pytest.approx(1.0, abs=1e-6)


def test_doppler_includes_listener_velocity():
    """Defect 2: a moving listener changes the ratio too, not just the source."""
    moving_listener = doppler_ratio(
        source_vel=[0.0, 0.0, 0.0],
        listener_vel=[0.0, 0.0, -5.0],
        unit_dir=[0.0, 0.0, -1.0],
        sound_speed=343.0,
    )
    stationary = doppler_ratio(
        source_vel=[0.0, 0.0, 0.0],
        listener_vel=[0.0, 0.0, 0.0],
        unit_dir=[0.0, 0.0, -1.0],
        sound_speed=343.0,
    )

    assert stationary == pytest.approx(1.0, abs=1e-6)
    assert moving_listener == pytest.approx(348.0 / 343.0, abs=1e-6)
    assert moving_listener > stationary


def test_azimuth_uses_atan2_for_left_and_right():
    """Defect 3: atan() cannot distinguish the left and right hemispheres."""
    right = azimuth_elevation(
        {"pos": [3.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0]}, LISTENER
    )
    left = azimuth_elevation(
        {"pos": [-3.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0]}, LISTENER
    )

    assert right[0] == pytest.approx(90.0, abs=1e-6)
    assert left[0] == pytest.approx(-90.0, abs=1e-6)
    assert right[1] == pytest.approx(0.0, abs=1e-6)
    assert left[1] == pytest.approx(0.0, abs=1e-6)


def test_pan_gains_51_are_normalized_at_every_azimuth():
    """Defect 4: 5.1 gains must sum to one instead of drifting with angle."""
    channels = {"C", "FL", "FR", "LFE", "RL", "RR"}

    for azimuth in (0.0, 15.0, 45.0, 90.0, 135.0, 180.0, -150.0):
        gains = pan_gains(azimuth, "5.1")

        assert set(gains) == channels
        assert gains["LFE"] == pytest.approx(0.0, abs=1e-6)
        assert all(gain >= 0.0 for gain in gains.values())
        assert sum(gains.values()) == pytest.approx(1.0, abs=1e-6)


def test_itd_varies_with_azimuth():
    """Defect 5: ITD must depend on azimuth, not be a fixed 0.6 ms."""
    front = itd_delay(0.0, 0.0875, 343.0)
    side = itd_delay(90.0, 0.0875, 343.0)
    other_side = itd_delay(-90.0, 0.0875, 343.0)

    assert front == pytest.approx(0.0, abs=1e-6)
    assert side == pytest.approx(0.0006558153894884939, abs=1e-6)
    assert other_side == pytest.approx(side, abs=1e-6)
    assert side > itd_delay(30.0, 0.0875, 343.0)
    assert front >= 0.0 and side >= 0.0 and other_side >= 0.0


def test_hrtf_select_uses_spherical_distance():
    """Defect 6: linear azimuth difference ignores elevation and picks wrong."""
    bank = [
        {"name": "front", "azimuth": 0.0, "elevation": 0.0},
        {"name": "rear_top", "azimuth": 180.0, "elevation": 90.0},
        {"name": "right", "azimuth": 90.0, "elevation": 0.0},
    ]

    assert hrtf_select(0.0, 80.0, bank) == "rear_top"
    assert hrtf_select(0.0, 0.0, bank) == "front"


def test_occlusion_lowpass_never_drops_below_floor():
    """Defect 7: a distant occluded source must stay audible above 200 Hz."""
    assert occlusion_lowpass(0.0, 10.0, 8000.0) == pytest.approx(8000.0, abs=1e-6)
    assert occlusion_lowpass(5.0, 10.0, 8000.0) == pytest.approx(4100.0, abs=1e-6)
    assert occlusion_lowpass(10.0, 10.0, 8000.0) == pytest.approx(200.0, abs=1e-6)
    assert occlusion_lowpass(100.0, 10.0, 8000.0) == pytest.approx(200.0, abs=1e-6)


def test_elevation_is_clamped_to_horizontal_hemisphere():
    """Defect 8: elevation beyond +/-90 must not flip a source behind the head."""
    azimuth, elevation = azimuth_elevation(
        {"pos": [0.0, 5.0, -5.0], "vel": [0.0, 0.0, 0.0]}, LISTENER
    )
    back_azimuth, back_elevation = azimuth_elevation(
        {"pos": [0.0, 5.0, 5.0], "vel": [0.0, 0.0, 0.0]}, LISTENER
    )
    _, overhead_elevation = azimuth_elevation(
        {"pos": [0.0, 5.0, 0.0], "vel": [0.0, 0.0, 0.0]}, LISTENER
    )

    assert azimuth == pytest.approx(0.0, abs=1e-6)
    assert elevation == pytest.approx(45.0, abs=1e-6)
    assert back_azimuth == pytest.approx(180.0, abs=1e-6)
    assert back_elevation == pytest.approx(45.0, abs=1e-6)
    assert overhead_elevation == pytest.approx(90.0, abs=1e-6)
    assert -90.0 <= elevation <= 90.0
    assert -90.0 <= back_elevation <= 90.0
    assert -90.0 <= overhead_elevation <= 90.0


def test_batch_rejects_source_without_hrtf_instead_of_skipping():
    """Defect 9: one source without an HRTF must abort the whole batch."""
    valid = {
        "id": "valid",
        "pos": [0.0, 0.0, -2.0],
        "vel": [0.0, 0.0, 0.0],
        "hrtf": "front",
    }
    missing_hrtf = {
        "id": "missing-hrtf",
        "pos": [2.0, 0.0, 0.0],
        "vel": [0.0, 0.0, 0.0],
    }

    with pytest.raises(SpatialError):
        spatialize_batch([valid, missing_hrtf], LISTENER, CONFIG)


def test_distance_gain_uses_metres_not_centimetres():
    """Defect 10: 4.0 means 4 m; a 100x unit mix-up must not change the gain."""
    assert distance_gain(4.0, "inverse", 1.0, 1.0) == pytest.approx(0.25, abs=1e-6)


# ---------------------------------------------------------------------------
# Additional invariants (5)
# ---------------------------------------------------------------------------


def test_reference_distance_linear_gain_is_unity():
    assert distance_gain(1.0, "linear", 1.0, 1.0) == pytest.approx(1.0, abs=1e-6)
    assert distance_gain(3.0, "linear", 3.0, 2.0) == pytest.approx(1.0, abs=1e-6)


def test_front_source_is_zero_azimuth_and_elevation():
    azimuth, elevation = azimuth_elevation(
        {"pos": [0.0, 0.0, -5.0], "vel": [0.0, 0.0, 0.0]}, LISTENER
    )

    assert azimuth == pytest.approx(0.0, abs=1e-6)
    assert elevation == pytest.approx(0.0, abs=1e-6)


def test_doppler_rejects_nonpositive_sound_speed():
    for sound_speed in (0.0, -343.0):
        with pytest.raises(SpatialError):
            doppler_ratio(
                source_vel=[0.0, 0.0, 0.0],
                listener_vel=[0.0, 0.0, 0.0],
                unit_dir=[0.0, 0.0, -1.0],
                sound_speed=sound_speed,
            )


def test_empty_source_list_returns_empty_batch():
    result = spatialize_batch([], LISTENER, CONFIG)

    assert result["sources"] == []
    assert result["layout"] == "5.1"


def test_spatialize_batch_preserves_input_order():
    sources = [
        {
            "id": "left",
            "pos": [-4.0, 0.0, 0.0],
            "vel": [0.0, 0.0, 0.0],
            "hrtf": "left",
        },
        {
            "id": "centre",
            "pos": [0.0, 0.0, -4.0],
            "vel": [0.0, 0.0, 0.0],
            "hrtf": "front",
        },
        {
            "id": "right",
            "pos": [4.0, 0.0, 0.0],
            "vel": [0.0, 0.0, 0.0],
            "hrtf": "right",
        },
    ]

    result = spatialize_batch(sources, LISTENER, CONFIG)

    assert [record["id"] for record in result["sources"]] == [
        "left",
        "centre",
        "right",
    ]
    assert [record["hrtf"] for record in result["sources"]] == [
        "left",
        "front",
        "right",
    ]
