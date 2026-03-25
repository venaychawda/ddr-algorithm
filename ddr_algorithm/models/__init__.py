"""
ddr_algorithm.models
--------------------
Data classes for DDR algorithm inputs and outputs.

The canonical definitions live in ddr_core.vehicle_signals (bundled in the
pre-compiled wheel). This package re-exports them so both import styles work:

    from ddr_algorithm.models import VehicleSignals          # via this __init__
    from ddr_algorithm.models.vehicle_signals import WheelSpeeds  # direct import
    from ddr_algorithm import VehicleSignals                  # via top-level __init__
"""

from ddr_algorithm.models.vehicle_signals import (
    DDROutput,
    DiagnosticWord,
    DriveDirection,
    GearPosition,
    SensorStatus,
    VehicleSignals,
    WheelSpeeds,
)

__all__ = [
    "VehicleSignals",
    "WheelSpeeds",
    "GearPosition",
    "DriveDirection",
    "DDROutput",
    "DiagnosticWord",
    "SensorStatus",
]
