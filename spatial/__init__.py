"""Public surface of the audio spatializer package.

Import the primitives from here (``from spatial import pan_gains``) or directly
from :mod:`spatial.panning`.  The implementation lives in
``spatial/panning.py``.
"""

from __future__ import annotations

from .panning import (
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

__all__ = [
    "SpatialError",
    "azimuth_elevation",
    "distance_gain",
    "doppler_ratio",
    "hrtf_select",
    "itd_delay",
    "occlusion_lowpass",
    "pan_gains",
    "spatialize_batch",
]
