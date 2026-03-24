"""
vehicle_signals.py — public re-export shim
-------------------------------------------
The canonical definitions live in ddr_core.vehicle_signals (bundled in the
pre-compiled wheel).  This module re-exports everything so that existing
user code continues to work unchanged:

    from ddr_algorithm.models.vehicle_signals import VehicleSignals  # still OK
    from ddr_core import VehicleSignals                               # also OK
"""

from ddr_core.vehicle_signals import (  # noqa: F401
    VehicleSignals,
    WheelSpeeds,
    GearPosition,
    DriveDirection,
    DDROutput,
    DiagnosticWord,
    SensorStatus,
)

__all__ = [
    "VehicleSignals", "WheelSpeeds", "GearPosition",
    "DriveDirection", "DDROutput", "DiagnosticWord", "SensorStatus",
]
