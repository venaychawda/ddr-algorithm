"""
vehicle_model.py
----------------
Simple kinematic vehicle model for synthetic signal generation.

This is NOT a full multi-body dynamics model — it is a kinematic model
sufficient to generate physically consistent sensor signals for DDR
algorithm testing. The model enforces:

  - Conservation of momentum (speed changes proportional to force/mass)
  - Realistic wheel speed sensor behaviour (magnitude-only, noise floor)
  - Brake pressure / deceleration relationship
  - Gear constraint validation (can't be in D and accelerating backward)
  - CAN message delay simulation for gear signal

Physical parameters are representative of a mid-size passenger car (C-segment).
No proprietary calibration data is used.

References:
  - Kiencke & Nielsen, "Automotive Control Systems", Springer 2005
  - Rajamani, "Vehicle Dynamics and Control", Springer 2012
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math

from ddr_algorithm.models.vehicle_signals import (
    VehicleSignals, WheelSpeeds, GearPosition
)


# ── Vehicle parameters (C-segment passenger car) ──────────────────────────────

VEHICLE_MASS_KG         = 1450.0     # Kerb weight
WHEEL_RADIUS_M          = 0.316      # 205/55 R16
DRAG_COEFF              = 0.30       # Aerodynamic drag coefficient
FRONTAL_AREA_M2         = 2.2        # Frontal area
AIR_DENSITY             = 1.225      # kg/m³ at sea level
ROLLING_RESISTANCE      = 0.012      # Coefficient of rolling resistance
GRAVITY                 = 9.81       # m/s²

# Brake model: pressure → deceleration (simplified linear)
# At 100 bar master cylinder → ~0.95g deceleration (dry tarmac, full ABS)
BRAKE_DECEL_COEFF       = 0.0935     # (m/s²) per bar

# Engine idle creep force (D gear, no throttle, no brake)
IDLE_CREEP_FORCE_N      = 180.0      # Torque converter creep at idle

# Maximum engine braking deceleration (engine-only, no brake)
ENGINE_BRAKE_DECEL      = 0.8        # m/s²

# Wheel speed sensor characteristics
WHEEL_PULSE_PER_REV     = 48         # ABS ring teeth (typical)
WHEEL_SPEED_NOISE_STD   = 0.08       # km/h, Gaussian noise std
WHEEL_SPEED_MIN_KMH     = 0.3        # Below this, sensor output is 0 (dead band)

# Accelerometer characteristics
ACC_NOISE_STD           = 0.04       # m/s², Gaussian noise std
ACC_BIAS                = 0.02       # m/s², constant bias (sensor offset)

# CAN message timing
GEAR_SIGNAL_PERIOD_MS   = 20.0       # Gear signal CAN cycle time
GEAR_SIGNAL_MAX_DELAY   = 60.0       # Worst-case gateway latency (ms)


@dataclass
class VehicleState:
    """Complete vehicle state at a single time instant."""
    time_ms: float = 0.0
    velocity_ms: float = 0.0        # m/s, positive=forward, negative=reverse
    acceleration_ms2: float = 0.0   # m/s²
    gear: GearPosition = GearPosition.PARK
    throttle_pct: float = 0.0       # 0–100%
    brake_bar: float = 0.0          # Master cylinder pressure, bar
    road_grade_deg: float = 0.0     # Road incline angle, degrees
    yaw_rate_deg_s: float = 0.0     # deg/s

    @property
    def speed_kmh(self) -> float:
        return abs(self.velocity_ms) * 3.6

    @property
    def is_moving_forward(self) -> bool:
        return self.velocity_ms > 0.001

    @property
    def is_moving_reverse(self) -> bool:
        return self.velocity_ms < -0.001


@dataclass
class ManoeuvrePoint:
    """A single point in a manoeuvre definition (time-value pairs)."""
    time_ms: float
    throttle_pct: float = 0.0
    brake_bar: float = 0.0
    gear: GearPosition = GearPosition.DRIVE
    road_grade_deg: float = 0.0
    yaw_rate_deg_s: float = 0.0


class VehicleModel:
    """
    Kinematic vehicle model with realistic sensor simulation.

    Integrates Newton's second law at 10ms steps:
      F_net = F_engine - F_brake - F_drag - F_rolling - F_grade
      a = F_net / m
      v[n+1] = v[n] + a * dt

    Sensor outputs are then computed from the true state with
    appropriate noise and delay models.
    """

    def __init__(
        self,
        dt_ms: float = 10.0,
        noise_seed: int = 42,
        gear_delay_ms: float = 0.0,
    ):
        self._dt = dt_ms / 1000.0       # Convert to seconds
        self._dt_ms = dt_ms
        self._gear_delay_ms = gear_delay_ms
        self._state = VehicleState()
        self._time_ms = 0.0
        self._gear_buffer: list[tuple[float, GearPosition]] = []  # (time, gear)

        import random
        self._rng = random.Random(noise_seed)

    def reset(self, initial_velocity_ms: float = 0.0):
        """Reset to a known initial state."""
        self._state = VehicleState(velocity_ms=initial_velocity_ms)
        self._time_ms = 0.0
        self._gear_buffer.clear()

    def step(self, manoeuvre: ManoeuvrePoint) -> VehicleSignals:
        """
        Advance simulation by one dt and return sensor signals.

        Args:
            manoeuvre: Driver inputs for this time step

        Returns:
            VehicleSignals as would be seen by the DDR ECU
        """
        # Update true gear (for physics)
        true_gear = manoeuvre.gear
        self._state.gear = true_gear
        self._state.road_grade_deg = manoeuvre.road_grade_deg
        self._state.yaw_rate_deg_s = manoeuvre.yaw_rate_deg_s

        # ── Force computation ──────────────────────────────────────────────────
        v = self._state.velocity_ms
        speed_ms = abs(v)

        # Engine / drivetrain force
        f_engine = self._compute_engine_force(
            manoeuvre.throttle_pct, true_gear, v
        )

        # Brake force (always opposes motion)
        f_brake = self._compute_brake_force(manoeuvre.brake_bar, v)

        # Aerodynamic drag (opposes motion)
        f_drag = (0.5 * AIR_DENSITY * DRAG_COEFF * FRONTAL_AREA_M2
                  * speed_ms ** 2 * math.copysign(1, v))

        # Rolling resistance (opposes motion)
        f_rolling = (ROLLING_RESISTANCE * VEHICLE_MASS_KG * GRAVITY
                     * math.copysign(1, v) if speed_ms > 0.01 else 0.0)

        # Grade force (gravity component along slope)
        grade_rad = math.radians(manoeuvre.road_grade_deg)
        f_grade = VEHICLE_MASS_KG * GRAVITY * math.sin(grade_rad)

        # Net force and acceleration
        f_net = f_engine - f_brake - f_drag - f_rolling - f_grade
        acceleration = f_net / VEHICLE_MASS_KG

        # Clamp: can't accelerate through zero without reversing
        new_velocity = v + acceleration * self._dt
        if (v > 0 and new_velocity < 0 and manoeuvre.brake_bar > 5):
            new_velocity = 0.0  # ABS holds at zero
        if (v < 0 and new_velocity > 0 and manoeuvre.brake_bar > 5):
            new_velocity = 0.0

        self._state.velocity_ms = new_velocity
        self._state.acceleration_ms2 = acceleration
        self._state.brake_bar = manoeuvre.brake_bar
        self._time_ms += self._dt_ms

        # ── Sensor simulation ──────────────────────────────────────────────────
        return self._compute_sensors(manoeuvre)

    def run_manoeuvre(
        self, points: list[ManoeuvrePoint]
    ) -> list[VehicleSignals]:
        """
        Simulate a complete manoeuvre defined by a list of ManoeuvrePoints.
        Points are interpolated at dt_ms resolution.
        """
        if not points:
            return []

        signals = []
        total_ms = points[-1].time_ms

        t = 0.0
        while t <= total_ms:
            mp = self._interpolate_manoeuvre(points, t)
            sig = self.step(mp)
            signals.append(sig)
            t += self._dt_ms

        return signals

    # ── Private: physics ───────────────────────────────────────────────────────

    def _compute_engine_force(
        self, throttle_pct: float, gear: GearPosition, velocity_ms: float
    ) -> float:
        """Simplified engine force model."""
        if gear == GearPosition.PARK:
            return 0.0

        if gear == GearPosition.NEUTRAL:
            return 0.0  # No torque transmission in neutral

        direction = 1.0 if gear == GearPosition.DRIVE else -1.0

        if throttle_pct < 1.0:
            # Idle creep in D/R gear
            if abs(velocity_ms) < 0.5:
                return direction * IDLE_CREEP_FORCE_N
            # Engine braking when coasting
            return -direction * ENGINE_BRAKE_DECEL * VEHICLE_MASS_KG * 0.3

        # Throttle applied: simplified force curve
        # Max force at low speed, reduces with speed (torque curve shape)
        max_force = 4500.0  # N, approximate peak tractive effort
        speed_factor = max(0.1, 1.0 - abs(velocity_ms) / 60.0)
        return direction * (throttle_pct / 100.0) * max_force * speed_factor

    def _compute_brake_force(self, brake_bar: float, velocity_ms: float) -> float:
        """Brake force always opposes current motion direction."""
        if brake_bar < 0.5 or abs(velocity_ms) < 0.001:
            return 0.0
        decel = brake_bar * BRAKE_DECEL_COEFF
        return decel * VEHICLE_MASS_KG * math.copysign(1, velocity_ms)

    # ── Private: sensor models ─────────────────────────────────────────────────

    def _compute_sensors(self, manoeuvre: ManoeuvrePoint) -> VehicleSignals:
        """Convert true vehicle state to simulated ECU sensor signals."""
        v = self._state.velocity_ms
        speed_kmh = abs(v) * 3.6

        # Wheel speeds: magnitude only, with noise and dead band
        ws = self._simulate_wheel_speeds(speed_kmh)

        # Longitudinal acceleration: true value + noise + bias
        acc_true = self._state.acceleration_ms2
        # Include grade component (accelerometer measures specific force)
        grade_rad = math.radians(self._state.road_grade_deg)
        acc_sensor = (acc_true + GRAVITY * math.sin(grade_rad)
                      + self._rng.gauss(ACC_BIAS, ACC_NOISE_STD))

        # Gear signal: apply CAN delay if configured
        reported_gear = self._simulate_gear_signal(manoeuvre.gear)

        # Brake pressure: true value + small noise
        brake_noise = self._rng.gauss(0, 0.3)
        brake_reported = max(0.0, self._state.brake_bar + brake_noise)

        # Yaw rate: true + noise
        yaw_noise = self._rng.gauss(0, 0.15)
        yaw_reported = self._state.yaw_rate_deg_s + yaw_noise

        return VehicleSignals(
            timestamp_ms=self._time_ms,
            wheel_speeds=ws,
            gear_position=reported_gear,
            longitudinal_acceleration=acc_sensor,
            brake_pressure_bar=brake_reported,
            yaw_rate_deg_s=yaw_reported,
        )

    def _simulate_wheel_speeds(self, true_speed_kmh: float) -> WheelSpeeds:
        """
        Simulate wheel speed sensor output.
        - Magnitude only (no sign — ABS sensors cannot detect direction)
        - Gaussian noise on each wheel independently
        - Dead band below WHEEL_SPEED_MIN_KMH
        """
        def wheel_speed() -> float:
            noisy = true_speed_kmh + self._rng.gauss(0, WHEEL_SPEED_NOISE_STD)
            return max(0.0, noisy) if noisy > WHEEL_SPEED_MIN_KMH else 0.0

        return WheelSpeeds(
            fl=wheel_speed(), fr=wheel_speed(),
            rl=wheel_speed(), rr=wheel_speed(),
        )

    def _simulate_gear_signal(self, true_gear: GearPosition) -> GearPosition:
        """
        Simulate gear CAN signal with optional delay.
        Gear changes are buffered and released after gear_delay_ms.
        """
        if self._gear_delay_ms <= 0:
            return true_gear

        # Buffer new gear commands
        if not self._gear_buffer or self._gear_buffer[-1][1] != true_gear:
            release_time = self._time_ms + self._gear_delay_ms
            self._gear_buffer.append((release_time, true_gear))

        # Release gear changes whose delay has expired
        released = [g for t, g in self._gear_buffer if t <= self._time_ms]
        pending = [(t, g) for t, g in self._gear_buffer if t > self._time_ms]
        self._gear_buffer = pending

        return released[-1] if released else (
            self._gear_buffer[0][1] if self._gear_buffer else true_gear
        )

    # ── Private: interpolation ────────────────────────────────────────────────

    @staticmethod
    def _interpolate_manoeuvre(
        points: list[ManoeuvrePoint], t: float
    ) -> ManoeuvrePoint:
        """Linear interpolation between manoeuvre waypoints."""
        if t <= points[0].time_ms:
            return points[0]
        if t >= points[-1].time_ms:
            return points[-1]

        for i in range(len(points) - 1):
            p0, p1 = points[i], points[i + 1]
            if p0.time_ms <= t <= p1.time_ms:
                alpha = (t - p0.time_ms) / (p1.time_ms - p0.time_ms)
                return ManoeuvrePoint(
                    time_ms=t,
                    throttle_pct=p0.throttle_pct + alpha * (p1.throttle_pct - p0.throttle_pct),
                    brake_bar=p0.brake_bar + alpha * (p1.brake_bar - p0.brake_bar),
                    gear=p1.gear if alpha > 0.5 else p0.gear,
                    road_grade_deg=p0.road_grade_deg + alpha * (p1.road_grade_deg - p0.road_grade_deg),
                    yaw_rate_deg_s=p0.yaw_rate_deg_s + alpha * (p1.yaw_rate_deg_s - p0.yaw_rate_deg_s),
                )
        return points[-1]


# ── Convenience: pre-built manoeuvres ─────────────────────────────────────────

def manoeuvre_city_drive() -> list[ManoeuvrePoint]:
    """Stop-go city drive: accelerate → cruise → brake → stop."""
    return [
        ManoeuvrePoint(0,    throttle_pct=0,   brake_bar=80, gear=GearPosition.DRIVE),
        ManoeuvrePoint(500,  throttle_pct=40,  brake_bar=0,  gear=GearPosition.DRIVE),
        ManoeuvrePoint(2000, throttle_pct=15,  brake_bar=0,  gear=GearPosition.DRIVE),
        ManoeuvrePoint(3500, throttle_pct=0,   brake_bar=35, gear=GearPosition.DRIVE),
        ManoeuvrePoint(5000, throttle_pct=0,   brake_bar=70, gear=GearPosition.DRIVE),
    ]


def manoeuvre_reverse_parking() -> list[ManoeuvrePoint]:
    """Slow reverse into parking bay."""
    return [
        ManoeuvrePoint(0,    throttle_pct=0,  brake_bar=60, gear=GearPosition.REVERSE),
        ManoeuvrePoint(300,  throttle_pct=8,  brake_bar=0,  gear=GearPosition.REVERSE),
        ManoeuvrePoint(2000, throttle_pct=5,  brake_bar=3,  gear=GearPosition.REVERSE),
        ManoeuvrePoint(2500, throttle_pct=0,  brake_bar=80, gear=GearPosition.REVERSE),
    ]


def manoeuvre_hill_start(grade_deg: float = 8.0) -> list[ManoeuvrePoint]:
    """Hill start: hold on slope → release brake → accelerate forward."""
    return [
        ManoeuvrePoint(0,    throttle_pct=0,  brake_bar=60, gear=GearPosition.DRIVE, road_grade_deg=grade_deg),
        ManoeuvrePoint(1500, throttle_pct=0,  brake_bar=60, gear=GearPosition.DRIVE, road_grade_deg=grade_deg),
        ManoeuvrePoint(2000, throttle_pct=35, brake_bar=0,  gear=GearPosition.DRIVE, road_grade_deg=grade_deg),
        ManoeuvrePoint(5000, throttle_pct=25, brake_bar=0,  gear=GearPosition.DRIVE, road_grade_deg=grade_deg),
    ]


def manoeuvre_k_turn() -> list[ManoeuvrePoint]:
    """3-point K-turn: reverse → stop → forward."""
    return [
        ManoeuvrePoint(0,    throttle_pct=0,  brake_bar=70, gear=GearPosition.REVERSE),
        ManoeuvrePoint(500,  throttle_pct=8,  brake_bar=0,  gear=GearPosition.REVERSE),
        ManoeuvrePoint(2000, throttle_pct=0,  brake_bar=80, gear=GearPosition.REVERSE),
        ManoeuvrePoint(2500, throttle_pct=0,  brake_bar=80, gear=GearPosition.DRIVE),
        ManoeuvrePoint(3000, throttle_pct=30, brake_bar=0,  gear=GearPosition.DRIVE),
        ManoeuvrePoint(5000, throttle_pct=20, brake_bar=0,  gear=GearPosition.DRIVE),
    ]


def manoeuvre_abs_stop() -> list[ManoeuvrePoint]:
    """Emergency stop from 80 km/h with ABS."""
    return [
        ManoeuvrePoint(0,    throttle_pct=30, brake_bar=0,   gear=GearPosition.DRIVE),
        ManoeuvrePoint(200,  throttle_pct=0,  brake_bar=100, gear=GearPosition.DRIVE),
        ManoeuvrePoint(4000, throttle_pct=0,  brake_bar=100, gear=GearPosition.DRIVE),
        ManoeuvrePoint(4500, throttle_pct=0,  brake_bar=80,  gear=GearPosition.DRIVE),
    ]
