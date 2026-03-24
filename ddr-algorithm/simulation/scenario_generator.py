"""
scenario_generator.py
---------------------
Synthetic driving scenario generator.

Generates realistic time-series VehicleSignals for 20 driving scenarios
with configurable noise models. This is the DDR algorithm's test bench —
since we cannot publish real vehicle data, all scenarios are synthetic
but modelled on realistic automotive sensor characteristics.

Noise model references:
  - Wheel speed sensor resolution: ~0.1 km/h at low speed
  - Accelerometer noise floor: ~0.05 m/s²
  - CAN message delay: up to 20ms per hop (3 hops typical)
"""

from __future__ import annotations
import random
import math
from dataclasses import dataclass
from typing import Iterator

from ddr_algorithm.models.vehicle_signals import (
    VehicleSignals, WheelSpeeds, GearPosition, DriveDirection
)


# ── Noise model parameters ─────────────────────────────────────────────────────

WHEEL_SPEED_NOISE_STD = 0.3      # km/h, standard deviation of Gaussian noise
ACC_NOISE_STD = 0.05             # m/s²
YAW_RATE_NOISE_STD = 0.2         # deg/s
BRAKE_NOISE_STD = 0.5            # bar


def _noisy(value: float, std: float, rng: random.Random) -> float:
    return max(0.0, value + rng.gauss(0, std))


def _wheel_speeds(
    speed: float,
    noise_std: float,
    fault_wheel: str | None,
    rng: random.Random,
) -> WheelSpeeds:
    """Generate four-corner wheel speeds with optional single-wheel fault."""
    speeds = {
        'fl': _noisy(speed, noise_std, rng),
        'fr': _noisy(speed, noise_std, rng),
        'rl': _noisy(speed, noise_std, rng),
        'rr': _noisy(speed, noise_std, rng),
    }
    if fault_wheel:
        speeds[fault_wheel] = 0.0   # Stuck-at-zero fault
    return WheelSpeeds(**speeds)


@dataclass
class ScenarioSpec:
    """Specification for a single synthetic driving scenario."""
    name: str
    expected_direction: DriveDirection
    description: str
    duration_ms: float              # Total scenario duration
    speed_profile: list[tuple[float, float]]   # (time_ms, speed_km/h)
    gear: GearPosition
    acc_profile: list[tuple[float, float]]     # (time_ms, acc_m/s²)
    brake_profile: list[tuple[float, float]]   # (time_ms, brake_bar)
    fault_wheel: str | None = None  # E.g. "fl" for FL sensor stuck-at-zero
    gear_delay_ms: float = 0.0      # Simulate stale CAN gear message


def _interpolate(profile: list[tuple[float, float]], t: float) -> float:
    """Linear interpolation over a time-value profile."""
    if not profile:
        return 0.0
    if t <= profile[0][0]:
        return profile[0][1]
    if t >= profile[-1][0]:
        return profile[-1][1]
    for i in range(len(profile) - 1):
        t0, v0 = profile[i]
        t1, v1 = profile[i + 1]
        if t0 <= t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return profile[-1][1]


class ScenarioGenerator:
    """
    Generates VehicleSignals time series from a ScenarioSpec.
    """

    def __init__(self, seed: int = 42, dt_ms: float = 10.0):
        self._rng = random.Random(seed)
        self._dt_ms = dt_ms

    def generate(self, spec: ScenarioSpec) -> list[VehicleSignals]:
        """Generate the full signal list for a scenario."""
        signals = []
        t = 0.0
        while t <= spec.duration_ms:
            speed = max(0.0, _interpolate(spec.speed_profile, t))
            acc = _interpolate(spec.acc_profile, t)
            brake = max(0.0, _interpolate(spec.brake_profile, t))

            # Apply gear delay: use previous gear for first gear_delay_ms
            gear = spec.gear
            if spec.gear_delay_ms > 0 and t < spec.gear_delay_ms:
                # Simulate stale CAN message — gear hasn't updated yet
                gear = GearPosition.REVERSE if spec.gear == GearPosition.DRIVE else GearPosition.DRIVE

            ws = _wheel_speeds(speed, WHEEL_SPEED_NOISE_STD, spec.fault_wheel, self._rng)

            signals.append(VehicleSignals(
                timestamp_ms=t,
                wheel_speeds=ws,
                gear_position=gear,
                longitudinal_acceleration=acc + self._rng.gauss(0, ACC_NOISE_STD),
                brake_pressure_bar=brake + self._rng.gauss(0, BRAKE_NOISE_STD),
                yaw_rate_deg_s=self._rng.gauss(0, YAW_RATE_NOISE_STD),
            ))
            t += self._dt_ms
        return signals


# ── Scenario library (20 scenarios) ───────────────────────────────────────────

def get_all_scenarios() -> list[ScenarioSpec]:
    return [
        # ── 1. Normal highway cruise ──────────────────────────────────────────
        ScenarioSpec(
            name="highway_cruise",
            expected_direction=DriveDirection.FORWARD,
            description="Sustained forward driving at 120 km/h",
            duration_ms=2000,
            speed_profile=[(0, 120), (2000, 120)],
            gear=GearPosition.DRIVE,
            acc_profile=[(0, 0.1), (2000, 0.1)],
            brake_profile=[(0, 0), (2000, 0)],
        ),
        # ── 2. City driving with stops ────────────────────────────────────────
        ScenarioSpec(
            name="city_stop_go",
            expected_direction=DriveDirection.STANDSTILL,
            description="City driving: accelerate, cruise, brake to stop (ends at standstill)",
            duration_ms=3000,
            speed_profile=[(0, 0), (500, 50), (1500, 50), (2500, 0), (3000, 0)],
            gear=GearPosition.DRIVE,
            acc_profile=[(0, 2.0), (500, 0.2), (1500, -2.5), (2500, 0), (3000, 0)],
            brake_profile=[(0, 0), (1500, 0), (2000, 40), (2500, 80), (3000, 80)],
        ),
        # ── 3. Normal reverse parking ─────────────────────────────────────────
        ScenarioSpec(
            name="reverse_parking",
            expected_direction=DriveDirection.REVERSE,
            description="Slow reverse into parking space",
            duration_ms=2000,
            speed_profile=[(0, 0), (300, 5), (1700, 5), (2000, 0)],
            gear=GearPosition.REVERSE,
            acc_profile=[(0, -0.3), (300, -0.1), (1700, -0.5), (2000, 0)],
            brake_profile=[(0, 5), (300, 0), (1700, 10), (2000, 60)],
        ),
        # ── 4. Complete standstill (traffic light) ────────────────────────────
        ScenarioSpec(
            name="standstill_traffic_light",
            expected_direction=DriveDirection.STANDSTILL,
            description="Stopped at red light, foot on brake",
            duration_ms=2000,
            speed_profile=[(0, 0), (2000, 0)],
            gear=GearPosition.DRIVE,
            acc_profile=[(0, 0), (2000, 0)],
            brake_profile=[(0, 80), (2000, 80)],
        ),
        # ── 5. EDGE: Gear mismatch (stale CAN) ───────────────────────────────
        ScenarioSpec(
            name="gear_mismatch_stale_can",
            expected_direction=DriveDirection.FORWARD,
            description="Moving forward but gear=R (stale CAN message)",
            duration_ms=1500,
            speed_profile=[(0, 15), (1500, 15)],
            gear=GearPosition.DRIVE,    # Will be correct after delay
            acc_profile=[(0, 0.8), (1500, 0.8)],
            brake_profile=[(0, 0), (1500, 0)],
            gear_delay_ms=800,          # Gear shows R for first 800ms
        ),
        # ── 6. EDGE: Rolling on hill (neutral, no brake) ──────────────────────
        ScenarioSpec(
            name="hill_rollback_neutral",
            expected_direction=DriveDirection.REVERSE,
            description="Rolling backwards on incline, gear=N, no brake",
            duration_ms=2000,
            speed_profile=[(0, 0), (500, 3), (2000, 6)],
            gear=GearPosition.NEUTRAL,
            acc_profile=[(0, -0.3), (2000, -0.5)],
            brake_profile=[(0, 0), (2000, 0)],
        ),
        # ── 7. EDGE: Wheel speed sensor fault (FL stuck at 0) ────────────────
        ScenarioSpec(
            name="wheel_sensor_fault_fl",
            expected_direction=DriveDirection.FORWARD,
            description="Driving forward, FL wheel sensor stuck at 0",
            duration_ms=2000,
            speed_profile=[(0, 35), (2000, 35)],
            gear=GearPosition.DRIVE,
            acc_profile=[(0, 0.5), (2000, 0.5)],
            brake_profile=[(0, 0), (2000, 0)],
            fault_wheel='fl',
        ),
        # ── 8. EDGE: K-turn mid-manoeuvre ────────────────────────────────────
        ScenarioSpec(
            name="k_turn_transition",
            expected_direction=DriveDirection.UNKNOWN,
            description="3-point turn: braking from reverse, shifting to Drive",
            duration_ms=1000,
            speed_profile=[(0, 4), (400, 0), (600, 0), (1000, 2)],
            gear=GearPosition.DRIVE,   # Just shifted from R to D
            acc_profile=[(0, -1.5), (400, 0), (600, 0.3), (1000, 0.8)],
            brake_profile=[(0, 60), (400, 80), (600, 20), (1000, 0)],
            gear_delay_ms=300,         # Gear=R for first 300ms
        ),
        # ── 9. EDGE: Low-speed creep in parking lot ───────────────────────────
        ScenarioSpec(
            name="low_speed_creep",
            expected_direction=DriveDirection.FORWARD,
            description="Creeping forward at <1 km/h in parking lot",
            duration_ms=3000,
            speed_profile=[(0, 0), (200, 0.8), (3000, 0.8)],
            gear=GearPosition.DRIVE,
            acc_profile=[(0, 0.05), (3000, 0.02)],
            brake_profile=[(0, 30), (200, 5), (3000, 5)],
        ),
        # ── 10. Hill start assist scenario ───────────────────────────────────
        ScenarioSpec(
            name="hill_start_forward",
            expected_direction=DriveDirection.FORWARD,
            description="HSA: holding on hill, then accelerating forward",
            duration_ms=3000,
            speed_profile=[(0, 0), (1000, 0), (1500, 10), (3000, 40)],
            gear=GearPosition.DRIVE,
            acc_profile=[(0, -0.2), (1000, 0), (1200, 1.5), (3000, 0.8)],
            brake_profile=[(0, 50), (1000, 50), (1200, 0), (3000, 0)],
        ),
        # ── 11. ABS braking event ─────────────────────────────────────────────
        ScenarioSpec(
            name="abs_braking",
            expected_direction=DriveDirection.STANDSTILL,
            description="Emergency braking from 80 km/h with ABS active (ends at standstill)",
            duration_ms=3000,
            speed_profile=[(0, 80), (2500, 0), (3000, 0)],
            gear=GearPosition.DRIVE,
            acc_profile=[(0, -8.0), (2500, 0), (3000, 0)],
            brake_profile=[(0, 0), (100, 100), (2500, 100), (3000, 80)],
        ),
        # ── 12. Long motorway cruise ──────────────────────────────────────────
        ScenarioSpec(
            name="motorway_cruise_long",
            expected_direction=DriveDirection.FORWARD,
            description="5-second motorway cruise at 130 km/h",
            duration_ms=5000,
            speed_profile=[(0, 130), (5000, 130)],
            gear=GearPosition.DRIVE,
            acc_profile=[(0, 0.05), (5000, 0.05)],
            brake_profile=[(0, 0), (5000, 0)],
        ),
        # ── 13. Parking space exit (forward) ──────────────────────────────────
        ScenarioSpec(
            name="parking_exit_forward",
            expected_direction=DriveDirection.FORWARD,
            description="Pulling out of parking space forward",
            duration_ms=2000,
            speed_profile=[(0, 0), (300, 8), (2000, 8)],
            gear=GearPosition.DRIVE,
            acc_profile=[(0, 1.0), (300, 0.2), (2000, 0.2)],
            brake_profile=[(0, 40), (300, 0), (2000, 0)],
        ),
        # ── 14. Uphill reverse ────────────────────────────────────────────────
        ScenarioSpec(
            name="uphill_reverse",
            expected_direction=DriveDirection.REVERSE,
            description="Reversing up a steep incline slowly",
            duration_ms=2000,
            speed_profile=[(0, 0), (400, 4), (2000, 4)],
            gear=GearPosition.REVERSE,
            acc_profile=[(0, -0.8), (2000, -0.5)],
            brake_profile=[(0, 20), (400, 5), (2000, 5)],
        ),
        # ── 15. Full stop then reverse ────────────────────────────────────────
        ScenarioSpec(
            name="stop_then_reverse",
            expected_direction=DriveDirection.REVERSE,
            description="Come to full stop, shift to R, then reverse",
            duration_ms=3000,
            speed_profile=[(0, 0), (1000, 0), (1500, 0), (2000, 6), (3000, 6)],
            gear=GearPosition.REVERSE,
            acc_profile=[(0, 0), (1500, 0), (1700, -0.4), (3000, -0.2)],
            brake_profile=[(0, 80), (1200, 80), (1500, 20), (1700, 0), (3000, 0)],
            gear_delay_ms=1200,     # Gear=D until 1200ms
        ),
        # ── 16. Gentle braking to stop ────────────────────────────────────────
        ScenarioSpec(
            name="gentle_brake_to_stop",
            expected_direction=DriveDirection.STANDSTILL,
            description="Slowing from 30 km/h to standstill",
            duration_ms=2000,
            speed_profile=[(0, 30), (1500, 0), (2000, 0)],
            gear=GearPosition.DRIVE,
            acc_profile=[(0, -2.0), (1500, 0), (2000, 0)],
            brake_profile=[(0, 0), (200, 25), (1500, 60), (2000, 60)],
        ),
        # ── 17. Park position ─────────────────────────────────────────────────
        ScenarioSpec(
            name="park_standstill",
            expected_direction=DriveDirection.STANDSTILL,
            description="Vehicle parked, gear=P, ignition on",
            duration_ms=2000,
            speed_profile=[(0, 0), (2000, 0)],
            gear=GearPosition.PARK,
            acc_profile=[(0, 0), (2000, 0)],
            brake_profile=[(0, 0), (2000, 0)],
        ),
        # ── 18. Multiple wheel faults ─────────────────────────────────────────
        ScenarioSpec(
            name="multiple_sensor_degradation",
            expected_direction=DriveDirection.FORWARD,
            description="Forward driving with FL sensor faulty",
            duration_ms=2000,
            speed_profile=[(0, 25), (2000, 25)],
            gear=GearPosition.DRIVE,
            acc_profile=[(0, 0.3), (2000, 0.3)],
            brake_profile=[(0, 0), (2000, 0)],
            fault_wheel='rl',
        ),
        # ── 19. Sport acceleration ────────────────────────────────────────────
        ScenarioSpec(
            name="sport_acceleration",
            expected_direction=DriveDirection.FORWARD,
            description="Hard forward acceleration 0-100 km/h",
            duration_ms=5000,
            speed_profile=[(0, 0), (5000, 100)],
            gear=GearPosition.DRIVE,
            acc_profile=[(0, 0.5), (200, 5.0), (4000, 3.0), (5000, 1.0)],
            brake_profile=[(0, 0), (5000, 0)],
        ),
        # ── 20. Slow reverse with brake pumping ───────────────────────────────
        ScenarioSpec(
            name="reverse_brake_pumping",
            expected_direction=DriveDirection.REVERSE,
            description="Reverse with intermittent brake application",
            duration_ms=3000,
            speed_profile=[(0, 0), (300, 6), (600, 3), (900, 6), (3000, 6)],
            gear=GearPosition.REVERSE,
            acc_profile=[(0, -0.4), (300, 0.3), (600, -0.4), (900, 0.3), (3000, 0)],
            brake_profile=[(0, 5), (300, 0), (500, 30), (700, 0), (3000, 0)],
        ),
    ]
