"""Core spatial-audio primitives for the audio-spatializer challenge.

Coordinate and unit conventions
-------------------------------
* Positions are Cartesian metres in a listener-aligned right-handed frame:
  ``+X`` is the listener's right, ``+Y`` is up and ``+Z`` is behind the
  listener, so straight ahead is ``-Z``.  Velocities are metres per second.
* Azimuth is measured in degrees from straight ahead, positive towards the
  right (``-180..180``); elevation is measured in degrees above the horizontal
  plane (``-90..90``).
* Time values are seconds.  Linear amplitude gains are non-negative floats.
* Floating-point assertions in the acceptance suite use an absolute tolerance
  of ``1e-6`` unless a docstring says otherwise.

Only the Python standard library is used; nothing here reads the clock, sleeps,
uses randomness, touches the network or writes files.
"""

from __future__ import annotations

import math


# A zero-distance source is clamped to this tiny positive distance (metres)
# before any division, so attenuation formulas stay finite instead of raising
# ZeroDivisionError or producing inf.
_MIN_DISTANCE = 1e-9

# Hard floor (hertz) for the occlusion low-pass cutoff.
_OCCLUSION_FLOOR_HZ = 200.0

# Doppler ratios are physically clamped to this symmetric interval.
_DOPPLER_MIN = 0.5
_DOPPLER_MAX = 2.0

# Speaker layouts: channel name -> azimuth in degrees (0 ahead, +90 right).
_LAYOUTS = {
    "mono": [("M", 0.0)],
    "stereo": [("L", -90.0), ("R", 90.0)],
    "quad": [("FL", -45.0), ("FR", 45.0), ("RL", -135.0), ("RR", 135.0)],
    "5.1": [
        ("C", 0.0),
        ("FL", -30.0),
        ("FR", 30.0),
        ("RL", -110.0),
        ("RR", 110.0),
    ],
}

# Channels that never carry directional panning energy.
_NON_DIRECTIONAL = {"5.1": ("LFE",)}

_DISTANCE_MODELS = ("linear", "inverse", "exponential")


class SpatialError(ValueError):
    """Raised when a spatial-audio argument violates its documented contract."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _finite(value: object, name: str) -> float:
    if not _is_finite_number(value):
        raise SpatialError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _vec3(value: object, name: str, *, unit: bool = False) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise SpatialError(f"{name} must be a 3-element sequence, got {value!r}")
    out = []
    for component in value:
        if not _is_finite_number(component):
            raise SpatialError(
                f"{name} components must be finite numbers, got {value!r}"
            )
        out.append(float(component))
    if unit:
        norm = math.sqrt(sum(component * component for component in out))
        if abs(norm - 1.0) > 1e-9:
            raise SpatialError(f"{name} must be a unit vector, got norm {norm}")
    return out


def _position_of(entity: object, name: str) -> list[float]:
    if not isinstance(entity, dict) or "pos" not in entity:
        raise SpatialError(f"{name} must be a dict containing a 'pos' vector")
    return _vec3(entity["pos"], f"{name}.pos")


def _wrap180(angle_deg: float) -> float:
    """Wrap an angle into ``(-180, 180]`` degrees."""
    wrapped = (angle_deg + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped


def _validate_bank(bank: object) -> tuple[list[tuple[str, float, float]], set[str]]:
    if not isinstance(bank, list) or not bank:
        raise SpatialError("hrtf_bank must be a non-empty list of entries")
    entries: list[tuple[str, float, float]] = []
    names: set[str] = set()
    for entry in bank:
        if not isinstance(entry, dict):
            raise SpatialError(f"each HRTF entry must be a dict, got {entry!r}")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise SpatialError(f"HRTF entry needs a non-empty name, got {entry!r}")
        if name in names:
            raise SpatialError(f"duplicated HRTF entry name: {name!r}")
        azimuth = _finite(entry.get("azimuth"), f"hrtf_bank[{name!r}].azimuth")
        elevation = _finite(entry.get("elevation"), f"hrtf_bank[{name!r}].elevation")
        if not -90.0 <= elevation <= 90.0:
            raise SpatialError(
                f"hrtf_bank[{name!r}].elevation must lie in [-90, 90]"
            )
        names.add(name)
        entries.append((name, azimuth, elevation))
    return entries, names


# ---------------------------------------------------------------------------
# Primitive spatial-audio functions
# ---------------------------------------------------------------------------


def distance_gain(
    distance: float,
    model: str,
    ref_distance: float,
    rolloff: float,
) -> float:
    """Compute a distance attenuation gain for a point source.

    ``distance`` and ``ref_distance`` are metres; both must be finite and
    ``ref_distance`` must be greater than zero.  A source closer than
    ``ref_distance`` is not boosted: the returned gain is capped at ``1.0``.
    A zero (or negative) ``distance`` is clamped to a tiny positive distance
    (``1e-9`` m) before any division, so the result is always finite instead of
    raising ``ZeroDivisionError`` or yielding ``inf``.

    Supported models (``rolloff >= 0`` is dimensionless):

    ``"linear"``
        ``gain = max(0, 1 - rolloff * (d - ref) / ref)`` for ``d > ref``.
    ``"inverse"``
        ``gain = ref / (ref + rolloff * (d - ref))`` for ``d > ref``.
    ``"exponential"``
        ``gain = (ref / d) ** rolloff`` for ``d > ref``.

    For ``d <= ref`` each model returns ``1.0``.  The result is clamped to
    ``[0.0, 1.0]``.  Distances are interpreted in metres, never centimetres.

    Args:
        distance: Source-to-listener distance in metres.
        model: One of ``"linear"``, ``"inverse"`` or ``"exponential"``.
        ref_distance: Reference distance in metres, strictly positive.
        rolloff: Non-negative attenuation factor.

    Returns:
        A finite linear gain in ``[0.0, 1.0]``.

    Raises:
        SpatialError: If an argument is non-finite, ``ref_distance <= 0``,
            ``rolloff < 0`` or ``model`` is unsupported.
    """
    distance = _finite(distance, "distance")
    ref_distance = _finite(ref_distance, "ref_distance")
    rolloff = _finite(rolloff, "rolloff")
    if ref_distance <= 0.0:
        raise SpatialError("ref_distance must be strictly positive")
    if rolloff < 0.0:
        raise SpatialError("rolloff must be non-negative")
    if model not in _DISTANCE_MODELS:
        raise SpatialError(f"unsupported distance model: {model!r}")

    # Clamp before any division so d=0 can never produce inf.
    d = max(distance, _MIN_DISTANCE)

    if d <= ref_distance:
        return 1.0

    if model == "linear":
        gain = 1.0 - rolloff * (d - ref_distance) / ref_distance
        gain = max(0.0, gain)
    elif model == "inverse":
        gain = ref_distance / (ref_distance + rolloff * (d - ref_distance))
    else:  # "exponential"
        gain = (ref_distance / d) ** rolloff

    return max(0.0, min(1.0, gain))


def doppler_ratio(
    source_vel: list[float],
    listener_vel: list[float],
    unit_dir: list[float],
    sound_speed: float,
) -> float:
    """Compute the Doppler frequency ratio for moving source and listener.

    ``unit_dir`` is the unit vector pointing from the listener towards the
    source.  The radial speeds are the projections ``v_s = source_vel ·
    unit_dir`` and ``v_l = listener_vel · unit_dir`` (metres per second), and
    the ratio is::

        ratio = (sound_speed + v_l) / (sound_speed + v_s)

    Moving the listener towards the source therefore raises the pitch, just as
    moving the source towards the listener does.  The ratio is clamped to
    ``[0.5, 2.0]`` after the division.  ``sound_speed`` must be a finite value
    greater than zero; engine code passes it explicitly rather than relying on
    a hidden temperature or unit constant.

    Args:
        source_vel: Source velocity ``[vx, vy, vz]`` in m/s.
        listener_vel: Listener velocity ``[vx, vy, vz]`` in m/s.
        unit_dir: Unit vector from listener to source.
        sound_speed: Speed of sound in m/s, strictly positive.

    Returns:
        The clamped, finite ratio ``(c + v_l) / (c + v_s)``.

    Raises:
        SpatialError: If ``sound_speed <= 0``, a value is non-finite, a vector
            is malformed, or ``unit_dir`` is not a unit vector.
    """
    source_vel = _vec3(source_vel, "source_vel")
    listener_vel = _vec3(listener_vel, "listener_vel")
    unit_dir = _vec3(unit_dir, "unit_dir", unit=True)
    sound_speed = _finite(sound_speed, "sound_speed")
    if sound_speed <= 0.0:
        raise SpatialError("sound_speed must be strictly positive")

    # Both velocities are projected onto the listener->source direction.
    v_source = sum(v * d for v, d in zip(source_vel, unit_dir))
    v_listener = sum(v * d for v, d in zip(listener_vel, unit_dir))

    ratio = (sound_speed + v_listener) / (sound_speed + v_source)
    return max(_DOPPLER_MIN, min(_DOPPLER_MAX, ratio))


def azimuth_elevation(source: dict, listener: dict) -> tuple[float, float]:
    """Return the source direction as ``(azimuth_deg, elevation_deg)``.

    Both ``source`` and ``listener`` are ``{"pos": [x, y, z], "vel": [x, y,
    z]}`` dictionaries; positions are metres.  The direction uses the
    right-handed frame documented at module level::

        dx, dy, dz = source.pos - listener.pos
        azimuth   = degrees(atan2(dx, -dz))
        elevation = degrees(atan2(dy, hypot(dx, dz)))

    ``atan2`` is required so that the left and right hemispheres are distinct.
    The elevation is measured above the horizontal plane and is clamped to
    ``[-90, 90]``; a source that is above and in front must not flip to the
    back hemisphere.  Azimuth lies in ``(-180, 180]``; a source directly
    behind the listener is ``180`` degrees, not ``0``.

    Args:
        source: Source dictionary with a three-element ``"pos"``.
        listener: Listener dictionary with a three-element ``"pos"``.

    Returns:
        ``(azimuth, elevation)`` in degrees.

    Raises:
        SpatialError: If either position is malformed or non-finite, or if the
            source coincides with the listener (the direction is undefined).
    """
    source_pos = _position_of(source, "source")
    listener_pos = _position_of(listener, "listener")

    dx = source_pos[0] - listener_pos[0]
    dy = source_pos[1] - listener_pos[1]
    dz = source_pos[2] - listener_pos[2]

    horizontal = math.hypot(dx, dz)
    if horizontal == 0.0 and dy == 0.0:
        raise SpatialError("source coincides with the listener; direction undefined")

    # atan2 keeps the left/right sign; atan would fold the two hemispheres.
    azimuth = math.degrees(math.atan2(dx, -dz))
    if azimuth == -180.0:
        azimuth = 180.0

    elevation = math.degrees(math.atan2(dy, horizontal))
    # Clamp so an out-of-range elevation can never flip behind the head.
    elevation = max(-90.0, min(90.0, elevation))

    return azimuth, elevation


def pan_gains(azimuth: float, layout: str) -> dict[str, float]:
    """Distribute a mono signal across the channels of ``layout``.

    The azimuth is in degrees (0 ahead, +90 right, ±180 behind) and is wrapped
    into ``(-180, 180]``.  Each layout uses circular angular interpolation
    between its two nearest speakers, followed by total-gain (L1) normalization
    so that ``sum(result.values()) == 1.0`` at every azimuth.  Without that
    normalization a 5.1 layout can swing by about 3 dB as the source moves
    between speaker axes.

    Layouts and channel keys:

    ``"mono"``
        ``{"M"}``; always ``{"M": 1.0}``.
    ``"stereo"``
        ``L`` at -90 degrees and ``R`` at +90 degrees.
    ``"quad"``
        ``FL`` -45, ``FR`` +45, ``RL`` -135 and ``RR`` +135 degrees.
    ``"5.1"``
        ``C`` 0, ``FL`` -30, ``FR`` +30, ``RL`` -110 and ``RR`` +110 degrees;
        ``LFE`` is non-directional and always receives ``0.0``.

    Every key for the requested layout is present, all gains are non-negative,
    and the gains sum to one within ``1e-6``.  Layout names are case
    insensitive.

    Args:
        azimuth: Source azimuth in degrees.
        layout: ``"mono"``, ``"stereo"``, ``"quad"`` or ``"5.1"``.

    Returns:
        A mapping from every channel name in ``layout`` to its linear gain.

    Raises:
        SpatialError: If ``azimuth`` is non-finite or ``layout`` is unknown.
    """
    azimuth = _finite(azimuth, "azimuth")
    if not isinstance(layout, str):
        raise SpatialError(f"layout must be a string, got {layout!r}")
    key = layout.lower()
    if key not in _LAYOUTS:
        raise SpatialError(f"unknown layout: {layout!r}")

    speakers = _LAYOUTS[key]
    gains: dict[str, float] = {name: 0.0 for name, _ in speakers}
    for extra in _NON_DIRECTIONAL.get(key, ()):
        gains[extra] = 0.0

    if key == "mono":
        gains["M"] = 1.0
        return gains

    az = _wrap180(azimuth)
    # Signed circular offset of each speaker from the source azimuth.
    signed = [
        (name, _wrap180(az - speaker_az)) for name, speaker_az in speakers
    ]
    nearest, second = sorted(signed, key=lambda item: abs(item[1]))[:2]
    name_a, delta_a = nearest
    name_b, delta_b = second

    if delta_a == 0.0 or (delta_a > 0.0) == (delta_b > 0.0):
        # Exactly on a speaker axis, or an antipodal boundary: the nearest
        # speaker takes everything and the other one is 180 degrees away.
        gains[name_a] = 1.0
        gains[name_b] = 0.0
    else:
        # Speakers bracket the azimuth; interpolate by circular arc distance.
        arc = abs(delta_a) + abs(delta_b)
        gains[name_a] = abs(delta_b) / arc
        gains[name_b] = abs(delta_a) / arc

    # L1 normalization: total directional energy is exactly one at every angle.
    directional_total = sum(
        gain for name, gain in gains.items()
        if name not in _NON_DIRECTIONAL.get(key, ())
    )
    if directional_total > 0.0:
        for name in gains:
            if name not in _NON_DIRECTIONAL.get(key, ()):
                gains[name] /= directional_total
    return gains


def itd_delay(azimuth: float, head_radius: float, sound_speed: float) -> float:
    """Return the interaural time difference (ITD) in seconds.

    Uses the Woodworth spherical-head approximation.  With
    ``theta = radians(min(abs(azimuth), 90))``::

        itd = (head_radius / sound_speed) * (theta + sin(theta))

    The delay is largest at the sides and zero at the front; because the
    azimuth is folded to its absolute lateral angle, the result is symmetric
    for left and right and is clamped at zero so it can never be negative.
    A fixed 0.6 ms delay for every direction is not acceptable: the returned
    value must vary with azimuth.  ``head_radius`` is in metres and
    ``sound_speed`` in metres per second; both must be positive.

    Args:
        azimuth: Source azimuth in degrees.
        head_radius: Effective head radius in metres, strictly positive.
        sound_speed: Speed of sound in m/s, strictly positive.

    Returns:
        Non-negative ITD in seconds.

    Raises:
        SpatialError: If an argument is non-finite, ``azimuth`` is outside a
            reasonable wrapped range, or a geometric parameter is not
            positive.
    """
    azimuth = _finite(azimuth, "azimuth")
    head_radius = _finite(head_radius, "head_radius")
    sound_speed = _finite(sound_speed, "sound_speed")
    if not -180.0 <= azimuth <= 180.0:
        raise SpatialError("azimuth must lie within the wrapped range [-180, 180]")
    if head_radius <= 0.0:
        raise SpatialError("head_radius must be strictly positive")
    if sound_speed <= 0.0:
        raise SpatialError("sound_speed must be strictly positive")

    lateral = min(abs(azimuth), 90.0)
    theta = math.radians(lateral)
    delay = (head_radius / sound_speed) * (theta + math.sin(theta))
    return max(0.0, delay)


def hrtf_select(azimuth: float, elevation: float, bank: list[dict]) -> str:
    """Select the nearest HRTF entry by spherical angular distance.

    ``bank`` is a non-empty list of dictionaries such as::

        {"name": "front", "azimuth": 0.0, "elevation": 0.0}

    Each entry needs a unique non-empty ``"name"`` plus finite ``"azimuth"``
    and ``"elevation"`` values.  Query and entry directions are converted to
    unit vectors and compared by great-circle angle::

        distance = acos(sin(el1)sin(el2)
                        + cos(el1)cos(el2)cos(az1 - az2))

    This is a spherical distance, not a linear difference of azimuths, so a
    high-elevation entry can win even when another entry has the same azimuth
    as the query.  Ties are resolved by the first matching entry in ``bank``.
    The selected entry's name is returned.  The query elevation is validated
    against ``[-90, 90]`` rather than silently wrapping behind the listener.

    Args:
        azimuth: Query azimuth in degrees.
        elevation: Query elevation in degrees, in ``[-90, 90]``.
        bank: Non-empty list of HRTF direction dictionaries.

    Returns:
        The ``"name"`` string of the nearest bank entry.

    Raises:
        SpatialError: If the query is non-finite/out of range, ``bank`` is
            empty or malformed, or entry names are duplicated.
    """
    azimuth = _finite(azimuth, "azimuth")
    elevation = _finite(elevation, "elevation")
    if not -90.0 <= elevation <= 90.0:
        raise SpatialError("elevation must lie in [-90, 90]")
    entries, _ = _validate_bank(bank)

    query_el = math.radians(elevation)
    query_az = math.radians(azimuth)
    sin_el1 = math.sin(query_el)
    cos_el1 = math.cos(query_el)

    best_name = ""
    best_distance = math.inf
    for name, entry_az, entry_el in entries:
        el2 = math.radians(entry_el)
        cos_angle = (
            sin_el1 * math.sin(el2)
            + cos_el1 * math.cos(el2) * math.cos(query_az - math.radians(entry_az))
        )
        # Guard against acos domain errors from floating-point drift.
        cos_angle = max(-1.0, min(1.0, cos_angle))
        distance = math.acos(cos_angle)
        # Strict comparison keeps the first entry on a tie.
        if distance < best_distance:
            best_distance = distance
            best_name = name
    return best_name


def occlusion_lowpass(
    distance: float,
    max_distance: float,
    cutoff_hz: float,
) -> float:
    """Return an occlusion low-pass cutoff in hertz.

    ``distance`` is the effective occlusion path length in metres and
    ``max_distance`` is the distance at which occlusion is maximal.  The cutoff
    falls linearly from ``cutoff_hz`` at zero distance to a hard floor of
    ``200 Hz`` at or beyond ``max_distance``::

        t      = clamp(distance / max_distance, 0, 1)
        cutoff = cutoff_hz - (cutoff_hz - 200) * t

    More occlusion therefore means a lower cutoff, but the result is never
    below 200 Hz, even for a source far outside the occluded zone.  Without the
    floor, distant sources would become completely mute.

    Args:
        distance: Occlusion distance in metres, non-negative.
        max_distance: Distance of maximum occlusion in metres, positive.
        cutoff_hz: Unoccluded cutoff in hertz, at least 200.

    Returns:
        A finite cutoff in ``[200, cutoff_hz]`` hertz.

    Raises:
        SpatialError: If values are non-finite/negative, ``max_distance <= 0``
            or ``cutoff_hz < 200``.
    """
    distance = _finite(distance, "distance")
    max_distance = _finite(max_distance, "max_distance")
    cutoff_hz = _finite(cutoff_hz, "cutoff_hz")
    if distance < 0.0:
        raise SpatialError("distance must be non-negative")
    if max_distance <= 0.0:
        raise SpatialError("max_distance must be strictly positive")
    if cutoff_hz < _OCCLUSION_FLOOR_HZ:
        raise SpatialError(
            f"cutoff_hz must be at least {_OCCLUSION_FLOOR_HZ:.0f} Hz"
        )

    t = max(0.0, min(1.0, distance / max_distance))
    cutoff = cutoff_hz - (cutoff_hz - _OCCLUSION_FLOOR_HZ) * t
    return max(_OCCLUSION_FLOOR_HZ, min(cutoff_hz, cutoff))


def spatialize_batch(
    sources: list[dict],
    listener: dict,
    config: dict,
) -> dict:
    """Spatialize a batch of sources atomically and preserve input order.

    Inputs
    ------
    ``sources`` is a list of dictionaries.  Each source must provide
    ``"pos": [x, y, z]`` (metres), ``"vel": [x, y, z]`` (m/s) and
    ``"hrtf": "<bank entry name>"``; an optional ``"id"`` string is copied to
    the output.  ``listener`` has ``"pos"`` and ``"vel"`` in the same frame.
    ``config`` must provide ``layout``, ``model``, ``ref_distance``,
    ``rolloff``, ``sound_speed``, ``head_radius``, ``max_occlusion_distance``,
    ``cutoff_hz`` and ``hrtf_bank``.

    Output
    ------
    A dictionary ``{"layout": <layout>, "sources": [record, ...]}``.  Records
    are in exactly the same order as ``sources`` and contain ``id``,
    ``azimuth``, ``elevation``, ``distance``, ``gain``, ``pan``, ``itd``,
    ``hrtf`` and ``lowpass_hz``.  For an empty ``sources`` list the result is
    ``{"layout": <layout>, "sources": []}``.

    Atomicity
    ---------
    Every source and its HRTF assignment is validated before any record is
    produced.  If any source is malformed, has no ``"hrtf"`` entry, names an
    unknown HRTF, or otherwise violates a contract, the whole call raises
    :class:`SpatialError`; it must never return a partial batch with the bad
    source silently skipped.  All calculations are pure, so a raised error
    leaves no externally visible partial state.

    Args:
        sources: Source dictionaries; an empty list is legal.
        listener: Listener dictionary with position and velocity.
        config: Spatializer configuration, including the HRTF bank.

    Returns:
        The layout plus ordered per-source spatialization records.

    Raises:
        SpatialError: If the listener, config, any source, or any HRTF
            assignment is invalid.
    """
    if not isinstance(config, dict):
        raise SpatialError("config must be a dictionary")
    required = (
        "layout",
        "model",
        "ref_distance",
        "rolloff",
        "sound_speed",
        "head_radius",
        "max_occlusion_distance",
        "cutoff_hz",
        "hrtf_bank",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise SpatialError(f"config is missing keys: {', '.join(missing)}")

    layout = config["layout"]
    if not isinstance(layout, str) or layout.lower() not in _LAYOUTS:
        raise SpatialError(f"unknown layout: {layout!r}")
    model = config["model"]
    if model not in _DISTANCE_MODELS:
        raise SpatialError(f"unsupported distance model: {model!r}")

    ref_distance = _finite(config["ref_distance"], "config.ref_distance")
    rolloff = _finite(config["rolloff"], "config.rolloff")
    sound_speed = _finite(config["sound_speed"], "config.sound_speed")
    head_radius = _finite(config["head_radius"], "config.head_radius")
    max_occlusion = _finite(
        config["max_occlusion_distance"], "config.max_occlusion_distance"
    )
    cutoff_hz = _finite(config["cutoff_hz"], "config.cutoff_hz")
    if ref_distance <= 0.0:
        raise SpatialError("config.ref_distance must be strictly positive")
    if rolloff < 0.0:
        raise SpatialError("config.rolloff must be non-negative")
    if sound_speed <= 0.0:
        raise SpatialError("config.sound_speed must be strictly positive")
    if head_radius <= 0.0:
        raise SpatialError("config.head_radius must be strictly positive")
    if max_occlusion <= 0.0:
        raise SpatialError(
            "config.max_occlusion_distance must be strictly positive"
        )
    if cutoff_hz < _OCCLUSION_FLOOR_HZ:
        raise SpatialError(
            f"config.cutoff_hz must be at least {_OCCLUSION_FLOOR_HZ:.0f} Hz"
        )

    _, bank_names = _validate_bank(config["hrtf_bank"])

    listener_pos = _position_of(listener, "listener")
    _vec3(listener.get("vel"), "listener.vel")

    if not isinstance(sources, list):
        raise SpatialError("sources must be a list")

    # ---- Validation pass: nothing is emitted until every source is legal. ----
    validated: list[tuple[dict, list[float], list[float], str]] = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise SpatialError(f"sources[{index}] must be a dictionary")
        pos = _vec3(source.get("pos"), f"sources[{index}].pos")
        vel = _vec3(source.get("vel"), f"sources[{index}].vel")
        hrtf = source.get("hrtf")
        if not isinstance(hrtf, str) or not hrtf:
            raise SpatialError(
                f"sources[{index}] is missing a valid 'hrtf' assignment"
            )
        if hrtf not in bank_names:
            raise SpatialError(
                f"sources[{index}] names an unknown HRTF: {hrtf!r}"
            )
        if pos == listener_pos:
            raise SpatialError(
                f"sources[{index}] coincides with the listener"
            )
        validated.append((source, pos, vel, hrtf))

    # ---- Computation pass: all inputs are now known to be valid. ----
    records = []
    for source, pos, _vel, hrtf in validated:
        azimuth, elevation = azimuth_elevation(source, listener)
        delta = [pos[i] - listener_pos[i] for i in range(3)]
        distance = math.sqrt(sum(component * component for component in delta))
        gain = distance_gain(distance, model, ref_distance, rolloff)
        records.append(
            {
                "id": source.get("id"),
                "azimuth": azimuth,
                "elevation": elevation,
                "distance": distance,
                "gain": gain,
                "pan": pan_gains(azimuth, layout),
                "itd": itd_delay(azimuth, head_radius, sound_speed),
                "hrtf": hrtf,
                "lowpass_hz": occlusion_lowpass(
                    distance, max_occlusion, cutoff_hz
                ),
            }
        )

    return {"layout": layout, "sources": records}
