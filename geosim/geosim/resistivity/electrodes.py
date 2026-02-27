"""Electrode array definitions for HIRT ERT system.

The HIRT uses ring electrodes embedded in borehole probes for
cross-hole electrical resistivity tomography (ERT).

Coordinate convention:
    Electrodes in borehole-local coordinates.
    Z-axis along borehole (positive = upward toward surface).
    X-axis perpendicular (distance between boreholes).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Electrode:
    """Single electrode.

    Parameters
    ----------
    position : ndarray, shape (3,)
        Electrode position [x, y, z] in meters.
    label : str
        Human-readable label (e.g., 'A1', 'B3').
    borehole : int
        Borehole index (0 or 1 for cross-hole).
    """

    position: np.ndarray
    label: str = ""
    borehole: int = 0

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)


@dataclass
class ElectrodeArray:
    """Array of electrodes for ERT measurements.

    Parameters
    ----------
    electrodes : list[Electrode]
        All electrodes in the array.
    name : str
        Array name/type (e.g., 'wenner', 'cross-hole').
    """

    electrodes: list[Electrode] = field(default_factory=list)
    name: str = ""

    @property
    def n_electrodes(self) -> int:
        """Total number of electrodes."""
        return len(self.electrodes)

    @property
    def positions(self) -> np.ndarray:
        """Electrode positions as (N, 3) array."""
        return np.array([e.position for e in self.electrodes])

    def borehole_electrodes(self, borehole: int) -> list[Electrode]:
        """Return electrodes belonging to a specific borehole."""
        return [e for e in self.electrodes if e.borehole == borehole]


def hirt_default_electrodes(
    n_rings: int = 8,
    ring_spacing: float = 0.10,
    borehole_separation: float = 0.50,
    start_depth: float = 0.10,
) -> ElectrodeArray:
    """Create the default HIRT cross-hole electrode configuration.

    Two boreholes with ring electrodes at regular intervals.

    Parameters
    ----------
    n_rings : int
        Number of ring electrodes per borehole.
    ring_spacing : float
        Vertical spacing between rings in meters.
    borehole_separation : float
        Horizontal distance between boreholes in meters.
    start_depth : float
        Depth of the first (shallowest) ring from probe bottom.

    Returns
    -------
    array : ElectrodeArray
        HIRT cross-hole electrode configuration.
    """
    electrodes = []

    for bh in range(2):
        x = bh * borehole_separation
        for i in range(n_rings):
            z = -(start_depth + i * ring_spacing)  # negative = below surface
            label = f"{'A' if bh == 0 else 'B'}{i + 1}"
            electrodes.append(Electrode(
                position=np.array([x, 0.0, z]),
                label=label,
                borehole=bh,
            ))

    return ElectrodeArray(electrodes=electrodes, name="hirt_crosshole")
