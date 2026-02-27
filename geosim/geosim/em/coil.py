"""Coil geometry definitions for HIRT MIT-3D instrument.

The HIRT uses a multi-frequency, multi-coil electromagnetic induction
(MIT) system mounted inside a borehole probe. This module defines the
coil configurations.

Coordinate convention:
    Coils are specified in probe-local coordinates.
    Z-axis runs along the probe (positive = upward toward surface).
    Origin at probe bottom.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CoilConfig:
    """Single coil configuration.

    Parameters
    ----------
    position : float
        Position along probe axis (meters from probe bottom).
    radius : float
        Coil radius in meters.
    turns : int
        Number of turns.
    role : str
        'tx' (transmitter) or 'rx' (receiver).
    label : str
        Human-readable label (e.g., 'TX1', 'RX_upper').
    """

    position: float
    radius: float
    turns: int = 1
    role: str = "tx"
    label: str = ""


@dataclass
class ProbeCoilSet:
    """Complete set of coils for a HIRT probe.

    Parameters
    ----------
    coils : list[CoilConfig]
        All coils in the probe.
    probe_length : float
        Total probe length in meters.
    probe_diameter : float
        Outer probe diameter in meters.
    """

    coils: list[CoilConfig] = field(default_factory=list)
    probe_length: float = 1.0
    probe_diameter: float = 0.05

    @property
    def transmitters(self) -> list[CoilConfig]:
        """Return transmitter coils."""
        return [c for c in self.coils if c.role == "tx"]

    @property
    def receivers(self) -> list[CoilConfig]:
        """Return receiver coils."""
        return [c for c in self.coils if c.role == "rx"]

    @property
    def n_coils(self) -> int:
        """Total number of coils."""
        return len(self.coils)


def hirt_default_coils() -> ProbeCoilSet:
    """Create the default HIRT MIT-3D coil configuration.

    Based on the HIRT design documents:
    - 3 transmitter coils at different heights
    - 3 receiver coils interleaved between transmitters
    - All coils wound on a 40mm diameter former
    - Probe length ~1.0m

    Returns
    -------
    coil_set : ProbeCoilSet
        HIRT default coil configuration.
    """
    coil_radius = 0.020  # 20mm radius (40mm diameter former)
    return ProbeCoilSet(
        coils=[
            CoilConfig(position=0.10, radius=coil_radius, turns=50, role="tx", label="TX1"),
            CoilConfig(position=0.25, radius=coil_radius, turns=100, role="rx", label="RX1"),
            CoilConfig(position=0.40, radius=coil_radius, turns=50, role="tx", label="TX2"),
            CoilConfig(position=0.55, radius=coil_radius, turns=100, role="rx", label="RX2"),
            CoilConfig(position=0.70, radius=coil_radius, turns=50, role="tx", label="TX3"),
            CoilConfig(position=0.85, radius=coil_radius, turns=100, role="rx", label="RX3"),
        ],
        probe_length=1.0,
        probe_diameter=0.04,
    )
