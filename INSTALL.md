# Installation

## 1. Clone the repository

```bash
git clone https://github.com/venaychawda/ddr-algorithm
cd ddr-algorithm
```

## 2. Install the core engine (pre-compiled)

The algorithm core is distributed as a pre-compiled wheel.
No source code is included in the wheel — only bytecode.

```bash
pip install wheels/ddr_core-1.0.0-py3-none-any.whl
```

## 3. Install the public package

```bash
pip install -e .
```

## 4. Run the test suite

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Expected: **58 passed**

## 5. Run the interactive demo

```bash
pip install -e ".[viz]"
streamlit run demo/streamlit_app.py
```

## 6. Open the notebook

```bash
pip install jupyter matplotlib pandas
jupyter notebook notebooks/ddr_explained.ipynb
```

---

## Why is the core pre-compiled?

The algorithm implementation (`signal_processor`, `plausibility`, `state_machine`,
`confidence`, `diagnostics`) is distributed as bytecode to protect the implementation
details. The public API (`DDREngine`, all data models, simulation tools, tests, and demo)
is fully open source under Apache 2.0.

The architecture, design decisions, and edge case handling are documented in full in
`notebooks/ddr_explained.ipynb` and `README.md`.
