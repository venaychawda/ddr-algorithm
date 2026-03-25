"""
ddr_algorithm — Drive Direction Recognition Algorithm
=====================================================
Public API package. The algorithm core (signal processor, plausibility engine,
state machine, confidence scorer, diagnostics) lives in ddr_core, distributed
as a pre-compiled wheel.

Typical usage:
    from ddr_algorithm import DDREngine, VehicleSignals, WheelSpeeds, GearPosition

    engine = DDREngine()
    output = engine.process(signals)
    print(output.direction.value, output.confidence)
"""

from ddr_algorithm.engine import DDREngine
from ddr_algorithm.models.vehicle_signals import (
    VehicleSignals,
    WheelSpeeds,
    GearPosition,
    DriveDirection,
    DDROutput,
    DiagnosticWord,
    SensorStatus,
)

__version__ = "1.0.0"

__all__ = [
    # Engine
    "DDREngine",
    # Input models
    "VehicleSignals",
    "WheelSpeeds",
    "GearPosition",
    "SensorStatus",
    # Output models
    "DriveDirection",
    "DDROutput",
    "DiagnosticWord",
]
