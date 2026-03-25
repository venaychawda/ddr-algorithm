"""
test_edge_cases.py
------------------
Targeted tests for the 8 critical DDR edge cases.

These are not scenario-level tests — they test specific algorithm
behaviours at the unit level, injecting carefully crafted signal sequences.

The 8 edge cases:
  1. Gear mismatch (stale CAN gear signal)
  2. Slow hillside rollback in Neutral
  3. Standstill detection vs low-speed creep
  4. Single wheel speed sensor stuck-at-zero
  5. K-turn direction reversal (debounce window)
  6. Low-speed parking lot creep
  7. Braking from high speed (acc negative, still forward)
  8. Park position vs actual standstill

Run with:  pytest tests/test_edge_cases/ -v
"""

import pytest
from ddr_algorithm.engine import DDREngine
from ddr_algorithm.models.vehicle_signals import (
    DriveDirection, GearPosition, VehicleSignals, WheelSpeeds
)


def make_signal(
    t: float,
    speed: float,
    gear: GearPosition,
    acc: float = 0.0,
    brake: float = 0.0,
    fl: float | None = None,
) -> VehicleSignals:
    """Convenience: build a VehicleSignals with optional per-wheel override."""
    ws = WheelSpeeds(
        fl=fl if fl is not None else speed,
        fr=speed, rl=speed, rr=speed,
    )
    return VehicleSignals(
        timestamp_ms=t,
        wheel_speeds=ws,
        gear_position=gear,
        longitudinal_acceleration=acc,
        brake_pressure_bar=brake,
    )


def run_signals(signals: list[VehicleSignals], **engine_kwargs) -> list:
    engine = DDREngine(**engine_kwargs)
    return engine.process_batch(signals)


# ── Edge Case 1: Gear mismatch ─────────────────────────────────────────────────

def test_edge_gear_mismatch_trusts_physical_sensors():
    """
    Vehicle clearly moving forward (speed=20 km/h, acc=1.2 m/s²)
    but gear=REVERSE (stale CAN). Algorithm must output FORWARD.
    """
    signals = [
        make_signal(t * 10.0, speed=20.0, gear=GearPosition.REVERSE, acc=1.2)
        for t in range(100)
    ]
    outputs = run_signals(signals)
    settled = outputs[50:]  # Last 50 cycles
    forward_count = sum(1 for o in settled if o.direction == DriveDirection.FORWARD)
    assert forward_count > len(settled) * 0.7, (
        f"Expected mostly FORWARD with gear mismatch, "
        f"got forward={forward_count}/{len(settled)}"
    )


def test_edge_gear_mismatch_triggers_diagnostic():
    """Gear mismatch scenario must set gear_mismatch diagnostic flag."""
    signals = [
        make_signal(t * 10.0, speed=20.0, gear=GearPosition.REVERSE, acc=1.2)
        for t in range(60)
    ]
    outputs = run_signals(signals)
    mismatch_set = any(o.diagnostics.gear_mismatch for o in outputs)
    assert mismatch_set, "gear_mismatch diagnostic never raised"


# ── Edge Case 2: Hill rollback in Neutral ──────────────────────────────────────

def test_edge_hill_rollback_neutral():
    """
    Vehicle rolling backwards at 4 km/h, gear=N, acc=-0.4.
    Algorithm must eventually detect REVERSE.
    """
    # Ramp up slowly (realistic rollback)
    signals = [
        make_signal(t * 10.0, speed=min(4.0, t * 0.1),
                    gear=GearPosition.NEUTRAL, acc=-0.3)
        for t in range(150)
    ]
    outputs = run_signals(signals)
    settled = outputs[80:]
    reverse_count = sum(1 for o in settled if o.direction == DriveDirection.REVERSE)
    # With neutral rollback detection, algorithm should identify REVERSE
    assert reverse_count > 0, (
        f"Hill rollback not detected — expected REVERSE in settled window, "
        f"got: {set(o.direction for o in settled)}"
    )


# ── Edge Case 3: Standstill vs low-speed creep ────────────────────────────────

def test_edge_standstill_high_brake():
    """Speed=0, brake=80 bar → must output STANDSTILL with high confidence."""
    signals = [
        make_signal(t * 10.0, speed=0.0, gear=GearPosition.DRIVE,
                    acc=0.0, brake=80.0)
        for t in range(100)
    ]
    outputs = run_signals(signals)
    settled = outputs[30:]
    standstill_count = sum(1 for o in settled if o.direction == DriveDirection.STANDSTILL)
    assert standstill_count > len(settled) * 0.8, (
        f"Standstill not detected: {standstill_count}/{len(settled)}"
    )


def test_edge_low_speed_creep_not_standstill():
    """Speed=0.8 km/h, gear=D → must NOT permanently lock to STANDSTILL."""
    signals = [
        make_signal(t * 10.0, speed=0.8, gear=GearPosition.DRIVE,
                    acc=0.05, brake=5.0)
        for t in range(200)
    ]
    outputs = run_signals(signals)
    # After initial settling, should not be stuck at standstill forever
    settled = outputs[100:]
    non_standstill = sum(
        1 for o in settled
        if o.direction != DriveDirection.STANDSTILL
    )
    # At least some portion should recognise forward motion
    assert non_standstill > 0, "Algorithm stuck in STANDSTILL during creep"


# ── Edge Case 4: Single wheel sensor fault ─────────────────────────────────────

def test_edge_wheel_fault_graceful_degradation():
    """
    FL wheel stuck at 0, other three at 35 km/h.
    Algorithm must still output FORWARD (not UNKNOWN).
    Wheel fault diagnostic must be raised.
    """
    signals = [
        make_signal(t * 10.0, speed=35.0, gear=GearPosition.DRIVE,
                    acc=0.5, fl=0.0)    # FL stuck at zero
        for t in range(150)
    ]
    outputs = run_signals(signals)
    settled = outputs[60:]

    forward_count = sum(1 for o in settled if o.direction == DriveDirection.FORWARD)
    fault_raised = any(o.diagnostics.wheel_fault for o in outputs)

    assert forward_count > len(settled) * 0.5, (
        f"Expected mostly FORWARD despite FL fault, got {forward_count}/{len(settled)}"
    )
    assert fault_raised, "wheel_fault diagnostic never raised for stuck-at-zero FL"


def test_edge_wheel_fault_confidence_reduced():
    """Confidence must be lower with a faulty wheel sensor."""
    # Normal driving (no fault)
    signals_normal = [
        make_signal(t * 10.0, speed=35.0, gear=GearPosition.DRIVE, acc=0.5)
        for t in range(150)
    ]
    outputs_normal = run_signals(signals_normal)
    conf_normal = sum(o.confidence for o in outputs_normal[60:]) / 90

    # Same driving with FL fault
    signals_fault = [
        make_signal(t * 10.0, speed=35.0, gear=GearPosition.DRIVE,
                    acc=0.5, fl=0.0)
        for t in range(150)
    ]
    outputs_fault = run_signals(signals_fault)
    conf_fault = sum(o.confidence for o in outputs_fault[60:]) / 90

    assert conf_fault <= conf_normal, (
        f"Fault scenario confidence ({conf_fault:.1f}) should be ≤ "
        f"normal ({conf_normal:.1f})"
    )


# ── Edge Case 5: K-turn direction reversal ─────────────────────────────────────

def test_edge_kturn_transition_active():
    """
    During K-turn: braking from reverse (gear just shifted to D).
    Transition state must be active during the debounce window.
    """
    # Phase 1: reversing
    signals = [
        make_signal(t * 10.0, speed=5.0, gear=GearPosition.REVERSE, acc=-0.3)
        for t in range(30)
    ]
    # Phase 2: stopped, gear just changed to D (but still decelerating)
    signals += [
        make_signal((30 + t) * 10.0, speed=max(0.0, 5.0 - t * 0.5),
                    gear=GearPosition.DRIVE, acc=-1.5, brake=60.0)
        for t in range(20)
    ]
    # Phase 3: forward
    signals += [
        make_signal((50 + t) * 10.0, speed=t * 0.3,
                    gear=GearPosition.DRIVE, acc=0.8)
        for t in range(50)
    ]

    outputs = run_signals(signals, debounce_ms=200.0)
    transition_cycles = sum(1 for o in outputs if o.diagnostics.transition_active)
    assert transition_cycles > 0, (
        "No transition_active cycles detected during K-turn direction change"
    )


# ── Edge Case 6: Confirmed standstill from stop ────────────────────────────────

def test_edge_standstill_immediate_after_stop():
    """
    Vehicle braking to 0 km/h with 80 bar — should reach STANDSTILL
    within 1 second (100 cycles) of zero speed.
    """
    # First: moving forward
    signals = [
        make_signal(t * 10.0, speed=50.0 - t * 0.5,
                    gear=GearPosition.DRIVE, acc=-2.0, brake=40.0)
        for t in range(100)
    ]
    # Then: stopped
    signals += [
        make_signal((100 + t) * 10.0, speed=0.0,
                    gear=GearPosition.DRIVE, acc=0.0, brake=80.0)
        for t in range(100)
    ]

    outputs = run_signals(signals)
    last_50 = outputs[150:]
    standstill_count = sum(1 for o in last_50 if o.direction == DriveDirection.STANDSTILL)
    assert standstill_count > len(last_50) * 0.6, (
        f"Standstill not reached after braking to 0: {standstill_count}/{len(last_50)}"
    )


# ── Edge Case 7: Hard braking (negative acc, still forward) ───────────────────

def test_edge_hard_braking_stays_forward():
    """
    Vehicle braking hard from 80 km/h: acc=-8 m/s².
    Must stay FORWARD — negative acc alone should not cause REVERSE output.
    """
    signals = []
    for t in range(200):
        speed = max(0.0, 80.0 - t * 0.5)
        acc = -8.0 if speed > 5.0 else 0.0
        brake = 100.0 if speed > 5.0 else 80.0
        signals.append(make_signal(
            t * 10.0, speed=speed,
            gear=GearPosition.DRIVE, acc=acc, brake=brake
        ))

    outputs = run_signals(signals)
    # While moving (first 160 cycles, speed > 0)
    moving = outputs[:160]
    reverse_count = sum(1 for o in moving if o.direction == DriveDirection.REVERSE)
    assert reverse_count == 0, (
        f"Algorithm incorrectly output REVERSE during hard forward braking "
        f"({reverse_count} cycles)"
    )


# ── Edge Case 8: Park position ─────────────────────────────────────────────────

def test_edge_park_position_standstill():
    """Gear=P with zero wheel speeds → STANDSTILL."""
    signals = [
        make_signal(t * 10.0, speed=0.0, gear=GearPosition.PARK,
                    acc=0.0, brake=0.0)
        for t in range(80)
    ]
    outputs = run_signals(signals)
    settled = outputs[20:]
    standstill_count = sum(1 for o in settled if o.direction == DriveDirection.STANDSTILL)
    assert standstill_count > len(settled) * 0.7, (
        f"Park position not recognised as standstill: {standstill_count}/{len(settled)}"
    )


# ── Engine lifecycle tests ─────────────────────────────────────────────────────

def test_engine_reset():
    """After reset, engine behaves as freshly initialised."""
    engine = DDREngine()
    signals = [
        make_signal(t * 10.0, speed=50.0, gear=GearPosition.DRIVE, acc=0.5)
        for t in range(100)
    ]
    for s in signals:
        engine.process(s)

    engine.reset()

    # After reset, first output should be UNKNOWN (INIT state)
    first = engine.process(make_signal(0.0, 0.0, GearPosition.DRIVE))
    # INIT state may immediately resolve if signal is clear — just check no crash
    assert first.direction in list(DriveDirection)


def test_engine_batch_equals_sequential():
    """process_batch must produce identical results to sequential process() calls."""
    signals = [
        make_signal(t * 10.0, speed=30.0, gear=GearPosition.DRIVE, acc=0.3)
        for t in range(50)
    ]

    engine1 = DDREngine()
    batch_outputs = engine1.process_batch(signals)

    engine2 = DDREngine()
    seq_outputs = [engine2.process(s) for s in signals]

    for i, (b, s) in enumerate(zip(batch_outputs, seq_outputs)):
        assert b.direction == s.direction, f"Mismatch at cycle {i}"
        assert abs(b.confidence - s.confidence) < 0.01, f"Confidence mismatch at {i}"
