# vehicle-ddr — Drive Direction Recognition Algorithm

[![CI](https://github.com/venaychawda/ddr-algorithm/actions/workflows/ci.yml/badge.svg)](https://github.com/venaychawda/ddr-algorithm/actions)
[![Tests](https://img.shields.io/badge/tests-58%20passed-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

A clean-room open-source implementation of the **Drive Direction Recognition (DDR)** algorithm used in automotive Vehicle Dynamics Control (VDC) and Electronic Stability Program (ESP) systems.

> **Background:** I was the Function Owner for Drive Direction Recognition at Bosch Global Software Technologies, responsible for this algorithm across all VDC platform projects in the India region — including ESP and Integrated Power Brake (IPB) systems. This repository is an independent, clean-room reimplementation based on standard automotive signal processing literature. No proprietary Bosch source code, calibration constants, or internal documents are used.

**[▶ Live Demo](https://venaychawda.github.io/ddr-algorithm/)** · **[📓 Notebook walkthrough](notebooks/ddr_explained.ipynb)**

---

## What is DDR?

Drive Direction Recognition determines whether a vehicle is moving **forward**, in **reverse**, or is at **standstill** — every 10ms. This is safety-critical:

- **ESP / ESC** applies brake force in a direction-dependent way. Wrong DDR → destabilises vehicle
- **HSA** (Hill Start Assist) must apply brake hold in the correct direction. Wrong DDR → rollback on hill
- **Park Assist / APA** operates at creep speeds where wheel sensors are unreliable
- **PRND interlocks** validate safe state before allowing gear changes

Under **ISO 26262**, DDR is typically classified **ASIL-B or ASIL-C** depending on vehicle architecture.

---

## Repository structure

```
ddr-algorithm/           ← this public repo
├── ddr_algorithm/
│   ├── engine.py        ← public API: DDREngine orchestrator
│   ├── models/
│   │   └── vehicle_signals.py   ← all data classes (open source)
│   └── simulation/
│       ├── scenario_generator.py  ← 20 synthetic scenarios
│       └── vehicle_model.py       ← kinematic vehicle model
├── wheels/
│   └── ddr_core-1.0.0-py3-none-any.whl   ← pre-compiled core (bytecode only)
├── tests/               ← 58 tests, fully open source
├── demo/
│   ├── streamlit_app.py          ← interactive demo
│   └── ddr_interactive_demo.html ← standalone browser demo
├── notebooks/
│   └── ddr_explained.ipynb       ← visual algorithm walkthrough
└── .github/workflows/ci.yml      ← CI on Python 3.10 / 3.11 / 3.12
```

> **Note on `wheels/ddr_core`:** The algorithm core (signal processor, plausibility engine,
> state machine, confidence scorer, diagnostics) is distributed as a pre-compiled wheel
> without source. The architecture, design decisions, and edge case handling are fully
> documented in `notebooks/ddr_explained.ipynb` and this README.

---

## Installation

```bash
git clone https://github.com/venaychawda/ddr-algorithm
cd ddr-algorithm

# Step 1: install the pre-compiled core
pip install wheels/ddr_core-1.0.0-py3-none-any.whl

# Step 2: install the public package
pip install -e .

# Step 3: run the tests
pip install -e ".[dev]"
pytest tests/ -v        # 58 passed
```

---

## Quick start

```python
from ddr_algorithm import DDREngine, VehicleSignals, WheelSpeeds, GearPosition

engine = DDREngine()

signals = VehicleSignals(
    timestamp_ms=0.0,
    wheel_speeds=WheelSpeeds(fl=50.0, fr=50.0, rl=50.0, rr=50.0),
    gear_position=GearPosition.DRIVE,
    longitudinal_acceleration=0.5,
    brake_pressure_bar=0.0,
)

output = engine.process(signals)
print(output.direction.value)   # FORWARD
print(output.confidence)        # 0–100%
print(output.diagnostics)       # DiagnosticWord
```

---

## Algorithm architecture — five layers

```
Raw sensors (10ms / 100 Hz)
        │
        ▼
┌───────────────────────────┐
│ L1 · Signal Processor     │  IIR filter · outlier rejection · sensor health
├───────────────────────────┤
│ L2 · Plausibility Engine  │  6 cross-signal checks · weighted votes
├───────────────────────────┤
│ L3 · State Machine (FSM)  │  FORWARD/REVERSE/STANDSTILL · 200ms debounce
├───────────────────────────┤
│ L4 · Confidence Scorer    │  Composite 0–100% · speed · health · time-in-state
├───────────────────────────┤
│ L5 · Diagnostics          │  8-bit word · CAN-encodable · session summary
└───────────────────────────┘
        │
        ▼
 DDROutput(direction, confidence, diagnostics)
```

Layers 1–5 are implemented in `ddr_core` (pre-compiled). `DDREngine` in `engine.py` orchestrates them and is fully open source.

---

## The 8 edge cases

| # | Scenario | Challenge | Approach |
|---|----------|-----------|----------|
| 1 | **Gear mismatch** | gear=R, vehicle moving forward (stale CAN) | Physical sensors override gear above 3 km/h |
| 2 | **Hill rollback in Neutral** | Wheel speeds magnitude-only, no gear | acc<0 + speed>0 + gear=N → REVERSE vote |
| 3 | **Standstill vs creep** | 0.8 km/h — sensor noise floor | Brake pressure bias for standstill |
| 4 | **Wheel sensor fault** | FL stuck at zero | Outlier rejection; continue on 3 healthy wheels |
| 5 | **K-turn** | Braking from reverse, gear just shifted to D | 200ms debounce → UNKNOWN during window |
| 6 | **Low-speed creep** | <1 km/h, poor SNR | Gear + brake delta weighted over wheel speed |
| 7 | **Hard braking** | acc=−8 m/s², still moving forward | Negative acc alone cannot vote REVERSE |
| 8 | **Park position** | gear=P, no brake | Gear=P maps directly to STANDSTILL |

---

## Diagnostic word

Every output cycle includes an 8-bit diagnostic word (CAN-encodable):

| Bit | Flag | Meaning |
|-----|------|---------|
| 0 | `gear_mismatch` | Gear signal contradicts physical sensors |
| 1 | `wheel_fault` | One or more wheel speed sensors FAULT |
| 2 | `low_speed_mode` | Below 0.5 km/h noise floor |
| 3 | `transition_active` | FSM in debounce transition window |
| 4 | `confidence_degraded` | Confidence below 70% |
| 5 | `plausibility_conflict` | 2+ checks in disagreement |

```python
from ddr_core import encode_diagnostic_word, decode_diagnostic_word

can_byte = encode_diagnostic_word(output.diagnostics)   # e.g. 0x05
diag     = decode_diagnostic_word(can_byte)
```

---

## Interactive demo

```bash
pip install -e ".[viz]"
streamlit run demo/streamlit_app.py
```

Or open `demo/ddr_interactive_demo.html` directly in any browser — no server needed.

---

## ISO 26262 relevance

- **Graceful degradation** on sensor fault — confidence reduced, output continues, `wheel_fault` flag set
- **No abrupt flips** — 200ms debounce prevents rapid FORWARD↔REVERSE transitions
- **Physical sensor priority** — gear CAN message can be delayed 60ms+ across gateway; physical sensors win above speed threshold
- **Explicit UNKNOWN state** — downstream safety systems hold last safe state during transition window
- **Diagnostic word** — enables DTC generation and bench-level monitoring

---

## Design notes: production experience

- Gear mismatch detection reflects a known CAN gateway latency failure mode in early ESP generations
- Neutral rollback detection using accelerometer sign is necessary because production wheel speed sensors (reluctance-type) output magnitude only — they cannot detect reverse from pulses alone
- The 200ms debounce is calibration-tuneable; in production it is set during vehicle validation at proving grounds
- Confidence as a separate output channel (not just direction enum) allows downstream consumers to implement their own fallback strategies

---

## License

Apache 2.0. See [LICENSE](LICENSE).

---

**Venay Chawda** — Software Architect, automotive embedded systems  
[LinkedIn](https://www.linkedin.com/in/venaychawda) · venaychawda@gmail.com
