"""
engine.py
---------
DDR Algorithm Engine — top-level orchestrator.

Wires together the five processing layers:
  1. SignalProcessor      — IIR filtering + outlier rejection
  2. PlausibilityEngine  — cross-signal consistency checks + votes
  3. DirectionStateMachine — FSM with hysteresis + debounce
  4. ConfidenceScorer    — composite confidence metric
  5. DiagnosticWord      — real-time flag emission

Usage:
    engine = DDREngine()
    for signals in data_stream:
        output = engine.process(signals)
        print(output.direction, output.confidence)
"""

from __future__ import annotations

from ddr_core import (
    ConfidenceScorer,
    DirectionStateMachine,
    PlausibilityEngine,
    SignalProcessor,
    aggregate_votes,
)

from ddr_algorithm.models.vehicle_signals import (
    DDROutput,
    DriveDirection,
    VehicleSignals,
)


class DDREngine:
    """
    Drive Direction Recognition Algorithm Engine.

    Thread-safety: Not thread-safe. Use one instance per task/thread.
    Cycle time: Designed for 10ms (100Hz). Configurable via dt_ms parameter.
    """

    def __init__(
        self,
        debounce_ms: float = 200.0,
        agree_cycles: int = 3,
    ):
        self._signal_processor = SignalProcessor()
        self._plausibility = PlausibilityEngine()
        self._fsm = DirectionStateMachine(
            debounce_ms=debounce_ms,
            agree_cycles=agree_cycles,
        )
        self._confidence_scorer = ConfidenceScorer()
        self._previous_direction = DriveDirection.UNKNOWN

    def process(
        self,
        signals: VehicleSignals,
        dt_ms: float = 10.0,
    ) -> DDROutput:
        """
        Process one cycle of vehicle signals.

        Args:
            signals: Raw sensor inputs for this cycle
            dt_ms:   Time elapsed since last call (default 10ms)

        Returns:
            DDROutput with direction, confidence, and diagnostics
        """
        # ── Layer 1: Signal pre-processing ────────────────────────────────────
        processed_wheels = self._signal_processor.process(signals)

        # ── Layer 2: Plausibility evaluation ──────────────────────────────────
        votes, diagnostics = self._plausibility.evaluate(signals, processed_wheels)
        proposed_direction, vote_confidence = aggregate_votes(votes)

        # ── Layer 3: State machine update ─────────────────────────────────────
        output_direction, transition_active = self._fsm.update(
            proposed_direction, vote_confidence, dt_ms
        )

        # Update transition diagnostic flag
        diagnostics.transition_active = transition_active

        # ── Layer 4: Confidence scoring ───────────────────────────────────────
        confidence = self._confidence_scorer.compute(
            vote_confidence=vote_confidence,
            mean_speed_kmh=processed_wheels.mean_speed(),
            processed_wheels=processed_wheels,
            fsm_state=self._fsm.current_state,
            time_in_state_ms=self._fsm.time_in_state_ms,
            transition_active=transition_active,
            diagnostics=diagnostics,
        )

        # ── Layer 5: Build output ─────────────────────────────────────────────
        output = DDROutput(
            timestamp_ms=signals.timestamp_ms,
            direction=output_direction,
            confidence=confidence,
            diagnostics=diagnostics,
            time_in_state_ms=self._fsm.time_in_state_ms,
            previous_direction=self._previous_direction,
        )

        self._previous_direction = output_direction
        return output

    def process_batch(
        self,
        signal_list: list[VehicleSignals],
        dt_ms: float = 10.0,
    ) -> list[DDROutput]:
        """Process a list of signals (e.g. from a recorded drive file)."""
        return [self.process(s, dt_ms) for s in signal_list]

    def reset(self):
        """Reset all internal state (ECU restart / ignition off)."""
        self._signal_processor.reset()
        self._fsm.reset()
        self._previous_direction = DriveDirection.UNKNOWN
