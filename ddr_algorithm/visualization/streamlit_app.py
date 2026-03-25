"""
streamlit_app.py
----------------
DDR Algorithm — Interactive Streamlit Demo

Provides two modes:
  1. Scenario explorer  — select a pre-built scenario, see DDR output in real time
  2. Upload CSV         — paste/upload your own signal data and run the algorithm

Run with:
    pip install vehicle-ddr[viz]
    streamlit run demo/streamlit_app.py

Deploy free at:
    streamlit.io/community/cloud → connect GitHub repo → set main file = demo/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ddr_algorithm.engine import DDREngine
from ddr_algorithm.models.vehicle_signals import (
    VehicleSignals, WheelSpeeds, GearPosition, DriveDirection
)
from ddr_algorithm.simulation.vehicle_model import (
    VehicleModel, manoeuvre_city_drive, manoeuvre_reverse_parking,
    manoeuvre_hill_start, manoeuvre_k_turn, manoeuvre_abs_stop,
)
from ddr_algorithm.simulation.scenario_generator import (
    ScenarioGenerator, get_all_scenarios
)
from ddr_algorithm.core.diagnostics import (
    SessionSummary, iter_diagnostic_events
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DDR Algorithm Demo",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ─────────────────────────────────────────────────────────────
DIR_COLORS = {
    DriveDirection.FORWARD:    "#1D9E75",
    DriveDirection.REVERSE:    "#D4900A",
    DriveDirection.STANDSTILL: "#7a7870",
    DriveDirection.UNKNOWN:    "#D85A30",
}

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e24;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-label { font-size: 11px; color: #7a7870; text-transform: uppercase; letter-spacing: .06em; }
    .metric-value { font-size: 28px; font-weight: 500; margin-top: 4px; }
    .diag-flag {
        display: inline-block;
        font-size: 11px;
        padding: 3px 10px;
        border-radius: 4px;
        margin: 2px;
        font-family: monospace;
    }
    .diag-on  { background: rgba(216,90,48,0.2); color: #F5C4B3; border: 1px solid #D85A30; }
    .diag-off { background: #1e1e24; color: #4a4845; border: 1px solid rgba(255,255,255,0.06); }
    .stTabs [data-baseweb="tab"] { font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 DDR Algorithm")
    st.markdown(
        "Drive Direction Recognition — open-source implementation.\n\n"
        "**Author:** Venay Chawda  \n"
        "**Background:** Function Owner for DDR @ Bosch Global Software Technologies\n\n"
        "---"
    )

    mode = st.radio(
        "Mode",
        ["🎬 Scenario explorer", "📁 Upload CSV", "🔧 Vehicle model"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Engine parameters**")
    debounce_ms = st.slider("Debounce window (ms)", 50, 500, 200, 50,
                            help="How long the FSM holds before committing to a new direction")
    agree_cycles = st.slider("Agreement cycles", 1, 10, 3, 1,
                             help="How many consecutive plausibility votes needed to start a transition")

    st.markdown("---")
    st.caption("github.com/venaychawda/ddr-algorithm")


# ── Helper: run engine on signal list ─────────────────────────────────────────
@st.cache_data
def run_engine(signal_tuples: list[tuple], debounce: float, agree: int):
    """Cached engine run. signal_tuples = list of (ts, wfl, wfr, wrl, wrr, acc, brake, gear_str)"""
    engine = DDREngine(debounce_ms=debounce, agree_cycles=agree)
    outputs = []
    for row in signal_tuples:
        ts, wfl, wfr, wrl, wrr, acc, brake, gear_str = row
        try:
            gear = GearPosition(gear_str)
        except ValueError:
            gear = GearPosition.UNKNOWN
        sig = VehicleSignals(
            timestamp_ms=ts,
            wheel_speeds=WheelSpeeds(fl=wfl, fr=wfr, rl=wrl, rr=wrr),
            gear_position=gear,
            longitudinal_acceleration=acc,
            brake_pressure_bar=brake,
        )
        outputs.append(engine.process(sig))
    return outputs


def outputs_to_df(signal_tuples, outputs) -> pd.DataFrame:
    """Build a combined DataFrame of inputs + outputs."""
    rows = []
    for (ts, wfl, wfr, wrl, wrr, acc, brake, gear), out in zip(signal_tuples, outputs):
        rows.append({
            "time_ms": ts,
            "whl_fl": wfl, "whl_fr": wfr, "whl_rl": wrl, "whl_rr": wrr,
            "mean_speed_kmh": (wfl + wfr + wrl + wrl) / 4,
            "long_acc_ms2": acc,
            "brake_bar": brake,
            "gear": gear,
            "direction": out.direction.value,
            "confidence": out.confidence,
            "gear_mismatch": out.diagnostics.gear_mismatch,
            "wheel_fault": out.diagnostics.wheel_fault,
            "low_speed_mode": out.diagnostics.low_speed_mode,
            "transition_active": out.diagnostics.transition_active,
            "confidence_degraded": out.diagnostics.confidence_degraded,
        })
    return pd.DataFrame(rows)


def build_plots(df: pd.DataFrame) -> go.Figure:
    """Build the main multi-panel Plotly figure."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.35, 0.25, 0.20, 0.20],
        vertical_spacing=0.04,
        subplot_titles=[
            "Drive Direction + Confidence",
            "Wheel Speeds (km/h)",
            "Longitudinal Acceleration (m/s²)",
            "Brake Pressure (bar)"
        ]
    )

    # ── Row 1: Direction bands + confidence ───────────────────────────────────
    dir_color_map = {
        "FORWARD": "#1D9E75", "REVERSE": "#D4900A",
        "STANDSTILL": "#7a7870", "UNKNOWN": "#D85A30"
    }

    # Coloured direction background bands
    prev_dir = None
    band_start = None
    for i, row in df.iterrows():
        d = row["direction"]
        if d != prev_dir:
            if prev_dir is not None:
                fig.add_vrect(
                    x0=band_start, x1=row["time_ms"],
                    fillcolor=dir_color_map.get(prev_dir, "#444"),
                    opacity=0.15, line_width=0, row=1, col=1
                )
            prev_dir = d
            band_start = row["time_ms"]
    if prev_dir is not None:
        fig.add_vrect(
            x0=band_start, x1=df["time_ms"].iloc[-1],
            fillcolor=dir_color_map.get(prev_dir, "#444"),
            opacity=0.15, line_width=0, row=1, col=1
        )

    # Confidence line
    fig.add_trace(go.Scatter(
        x=df["time_ms"], y=df["confidence"],
        mode="lines", name="Confidence %",
        line=dict(color="#378ADD", width=2),
    ), row=1, col=1)

    # Direction as text annotations at transitions
    transitions = df[df["direction"] != df["direction"].shift()]
    for _, tr in transitions.iterrows():
        fig.add_annotation(
            x=tr["time_ms"], y=95,
            text=tr["direction"][:3],
            font=dict(size=9, color=dir_color_map.get(tr["direction"], "#fff")),
            showarrow=False, row=1, col=1
        )

    # Gear mismatch markers
    mismatch = df[df["gear_mismatch"]]
    if not mismatch.empty:
        fig.add_trace(go.Scatter(
            x=mismatch["time_ms"], y=[105]*len(mismatch),
            mode="markers", name="gear_mismatch",
            marker=dict(color="#D85A30", size=6, symbol="triangle-down"),
        ), row=1, col=1)

    fig.update_yaxes(title_text="Confidence %", range=[0, 110], row=1, col=1)

    # ── Row 2: Wheel speeds ────────────────────────────────────────────────────
    colors = ["#378ADD", "#7F77DD", "#1D9E75", "#D4900A"]
    for col_name, color, label in zip(
        ["whl_fl", "whl_fr", "whl_rl", "whl_rr"],
        colors,
        ["FL", "FR", "RL", "RR"]
    ):
        fig.add_trace(go.Scatter(
            x=df["time_ms"], y=df[col_name],
            mode="lines", name=f"Speed {label}",
            line=dict(color=color, width=1),
        ), row=2, col=1)

    # Wheel fault markers
    fault = df[df["wheel_fault"]]
    if not fault.empty:
        fig.add_trace(go.Scatter(
            x=fault["time_ms"], y=fault["whl_fl"],
            mode="markers", name="wheel_fault",
            marker=dict(color="#D85A30", size=4, symbol="x"),
            showlegend=True
        ), row=2, col=1)

    fig.update_yaxes(title_text="km/h", row=2, col=1)

    # ── Row 3: Acceleration ────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["time_ms"], y=df["long_acc_ms2"],
        mode="lines", name="Long. acc",
        line=dict(color="#9FE1CB", width=1.5),
        fill="tozeroy", fillcolor="rgba(29,158,117,0.08)"
    ), row=3, col=1)
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.15)", width=1), row=3, col=1)
    fig.update_yaxes(title_text="m/s²", row=3, col=1)

    # ── Row 4: Brake pressure ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["time_ms"], y=df["brake_bar"],
        mode="lines", name="Brake",
        line=dict(color="#E24B4A", width=1.5),
        fill="tozeroy", fillcolor="rgba(226,75,74,0.08)"
    ), row=4, col=1)
    fig.update_yaxes(title_text="bar", row=4, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=4, col=1)

    # ── Transition highlights ──────────────────────────────────────────────────
    trans = df[df["transition_active"]]
    if not trans.empty:
        for _, tr_row in trans.iterrows():
            fig.add_vrect(
                x0=tr_row["time_ms"] - 5,
                x1=tr_row["time_ms"] + 5,
                fillcolor="rgba(255,255,255,0.04)",
                line_width=0,
            )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        height=600,
        paper_bgcolor="#0d0d10",
        plot_bgcolor="#0d0d10",
        font=dict(family="DM Sans, sans-serif", color="#e4e2db", size=11),
        legend=dict(
            orientation="h", x=0, y=-0.08,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10)
        ),
        margin=dict(l=50, r=20, t=30, b=40),
        hovermode="x unified",
    )
    for i in range(1, 5):
        fig.update_xaxes(
            showgrid=True, gridcolor="rgba(255,255,255,0.06)",
            zeroline=False, row=i, col=1
        )
        fig.update_yaxes(
            showgrid=True, gridcolor="rgba(255,255,255,0.06)",
            zeroline=False, row=i, col=1
        )

    return fig


def render_metrics(outputs, df):
    """Top metrics row."""
    summary = SessionSummary.from_outputs(outputs)
    last = outputs[-1]
    total = len(outputs)

    col1, col2, col3, col4, col5 = st.columns(5)

    dir_color = {
        "FORWARD": "#1D9E75", "REVERSE": "#D4900A",
        "STANDSTILL": "#7a7870", "UNKNOWN": "#D85A30"
    }
    d = last.direction.value

    with col1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Final direction</div>
          <div class="metric-value" style="color:{dir_color.get(d,'#fff')}">{d}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Confidence</div>
          <div class="metric-value">{last.confidence:.0f}%</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Forward</div>
          <div class="metric-value" style="color:#1D9E75">{100*summary.forward_cycles/total:.0f}%</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Reverse</div>
          <div class="metric-value" style="color:#D4900A">{100*summary.reverse_cycles/total:.0f}%</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Transitions</div>
          <div class="metric-value">{summary.transition_events}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")  # Spacer


def render_diagnostics(outputs):
    """Diagnostic flags for last cycle."""
    last = outputs[-1].diagnostics
    flags = {
        "gear_mismatch [bit 0]": last.gear_mismatch,
        "wheel_fault [bit 1]": last.wheel_fault,
        "low_speed_mode [bit 2]": last.low_speed_mode,
        "transition_active [bit 3]": last.transition_active,
        "confidence_degraded [bit 4]": last.confidence_degraded,
        "plausibility_conflict [bit 5]": last.plausibility_conflict,
    }
    html = ""
    byte = 0
    for i, (name, active) in enumerate(flags.items()):
        cls = "diag-on" if active else "diag-off"
        html += f'<span class="diag-flag {cls}">{name}</span>'
        if active:
            byte |= (1 << i)

    st.markdown(
        f'<div>{html}</div>'
        f'<div style="font-family:monospace;font-size:11px;color:#7a7870;margin-top:6px">'
        f'CAN byte: <span style="color:#9FE1CB">0x{byte:02X}</span>&nbsp;·&nbsp;'
        f'binary: <span style="color:#9FE1CB">{byte:08b}</span></div>',
        unsafe_allow_html=True
    )


def render_event_log(outputs):
    """Show diagnostic event log."""
    events = list(iter_diagnostic_events(outputs))
    if not events:
        st.caption("No diagnostic events in this session.")
        return
    event_df = pd.DataFrame(events, columns=["time_ms", "event"])
    event_df["time_ms"] = event_df["time_ms"].apply(lambda x: f"{x:.0f} ms")
    st.dataframe(event_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1: Scenario explorer
# ═══════════════════════════════════════════════════════════════════════════════

if mode == "🎬 Scenario explorer":
    st.markdown("## Scenario Explorer")
    st.markdown(
        "Select a pre-built driving scenario. The algorithm runs at 100Hz (10ms cycles) "
        "on synthetically generated signals with realistic sensor noise."
    )

    scenarios = get_all_scenarios()
    sc_names = {s.name: s for s in scenarios}

    col_sel, col_desc = st.columns([1, 2])
    with col_sel:
        selected_name = st.selectbox(
            "Scenario",
            list(sc_names.keys()),
            format_func=lambda x: x.replace("_", " ").title()
        )
    with col_desc:
        sc = sc_names[selected_name]
        st.info(f"**{sc.description}**  \n"
                f"Duration: {sc.duration_ms:.0f} ms  ·  "
                f"Expected: {sc.expected_direction.value}  ·  "
                f"Gear: {sc.gear.value}"
                + (f"  ·  ⚠️ Fault wheel: {sc.fault_wheel}" if sc.fault_wheel else "")
                + (f"  ·  ⏱ Gear delay: {sc.gear_delay_ms:.0f}ms" if sc.gear_delay_ms else ""))

    # Generate signals
    gen = ScenarioGenerator(seed=42)
    signals = gen.generate(sc)
    signal_tuples = [
        (
            s.timestamp_ms,
            s.wheel_speeds.fl, s.wheel_speeds.fr,
            s.wheel_speeds.rl, s.wheel_speeds.rr,
            s.longitudinal_acceleration,
            s.brake_pressure_bar,
            s.gear_position.value,
        )
        for s in signals
    ]

    outputs = run_engine(tuple(signal_tuples), debounce_ms, agree_cycles)
    df = outputs_to_df(signal_tuples, outputs)

    st.markdown("---")
    render_metrics(outputs, df)

    tabs = st.tabs(["📈 Timeline", "🔬 Diagnostics", "📋 Event log", "📥 Export"])

    with tabs[0]:
        st.plotly_chart(build_plots(df), use_container_width=True)

    with tabs[1]:
        st.markdown("**Last cycle diagnostic word**")
        render_diagnostics(outputs)
        st.markdown("")
        st.markdown("**Diagnostic flag timeline**")
        diag_df = df[["time_ms", "gear_mismatch", "wheel_fault",
                       "low_speed_mode", "transition_active", "confidence_degraded"]].copy()
        diag_df = diag_df[diag_df[["gear_mismatch","wheel_fault","low_speed_mode",
                                    "transition_active","confidence_degraded"]].any(axis=1)]
        if diag_df.empty:
            st.caption("No diagnostic flags active in this scenario.")
        else:
            st.dataframe(diag_df, use_container_width=True, hide_index=True)

    with tabs[2]:
        render_event_log(outputs)

    with tabs[3]:
        csv_data = df.to_csv(index=False)
        st.download_button(
            "⬇ Download full session CSV",
            csv_data,
            file_name=f"ddr_{selected_name}.csv",
            mime="text/csv"
        )
        st.markdown("**CSV columns:**")
        st.code(", ".join(df.columns.tolist()))


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 2: Upload CSV
# ═══════════════════════════════════════════════════════════════════════════════

elif mode == "📁 Upload CSV":
    st.markdown("## Upload your own signal data")
    st.markdown(
        "Upload a CSV with vehicle sensor signals. "
        "The DDR algorithm will process each row as a 10ms cycle."
    )

    st.markdown("**Required columns:**")
    st.code("timestamp_ms, whl_fl, whl_fr, whl_rl, whl_rr, long_acc_ms2, brake_bar, gear")
    st.caption("gear values: P, R, N, D  ·  speeds in km/h  ·  acc in m/s²  ·  brake in bar")

    # Sample data generator
    with st.expander("📋 Generate sample CSV"):
        sample_scenario = st.selectbox("Base scenario", ["city_drive", "reverse_parking", "k_turn"])
        model_sample = VehicleModel(noise_seed=99)
        manoeuvre_map = {
            "city_drive": manoeuvre_city_drive(),
            "reverse_parking": manoeuvre_reverse_parking(),
            "k_turn": manoeuvre_k_turn(),
        }
        sample_signals = model_sample.run_manoeuvre(manoeuvre_map[sample_scenario])
        rows = []
        for s in sample_signals:
            rows.append({
                "timestamp_ms": s.timestamp_ms,
                "whl_fl": round(s.wheel_speeds.fl, 2),
                "whl_fr": round(s.wheel_speeds.fr, 2),
                "whl_rl": round(s.wheel_speeds.rl, 2),
                "whl_rr": round(s.wheel_speeds.rr, 2),
                "long_acc_ms2": round(s.longitudinal_acceleration, 3),
                "brake_bar": round(s.brake_pressure_bar, 1),
                "gear": s.gear_position.value,
            })
        sample_df = pd.DataFrame(rows)
        st.dataframe(sample_df.head(10), use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download sample CSV",
            sample_df.to_csv(index=False),
            file_name=f"sample_{sample_scenario}.csv",
            mime="text/csv"
        )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
            required = {"timestamp_ms", "whl_fl", "whl_fr", "whl_rl", "whl_rr",
                        "long_acc_ms2", "brake_bar", "gear"}
            missing = required - set(df_raw.columns)
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                signal_tuples = [
                    (row.timestamp_ms, row.whl_fl, row.whl_fr, row.whl_rl, row.whl_rr,
                     row.long_acc_ms2, row.brake_bar, str(row.gear))
                    for _, row in df_raw.iterrows()
                ]
                outputs = run_engine(tuple(signal_tuples), debounce_ms, agree_cycles)
                df_out = outputs_to_df(signal_tuples, outputs)

                st.success(f"✅ Processed {len(outputs)} cycles")
                st.markdown("---")
                render_metrics(outputs, df_out)
                st.plotly_chart(build_plots(df_out), use_container_width=True)
                st.markdown("**Diagnostic word (last cycle)**")
                render_diagnostics(outputs)
                st.markdown("---")
                st.download_button(
                    "⬇ Download results CSV",
                    df_out.to_csv(index=False),
                    file_name="ddr_results.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 3: Vehicle model explorer
# ═══════════════════════════════════════════════════════════════════════════════

elif mode == "🔧 Vehicle model":
    st.markdown("## Vehicle Model Explorer")
    st.markdown(
        "Simulate real driving manoeuvres using the kinematic vehicle model. "
        "Observe how physical parameters (throttle, brake, grade, gear delay) "
        "affect DDR output."
    )

    col_m, col_p = st.columns([1, 2])

    with col_m:
        manoeuvre_name = st.selectbox("Manoeuvre", [
            "City drive", "Reverse parking",
            "Hill start", "K-turn", "ABS emergency stop"
        ])
        grade_deg = 0.0
        if manoeuvre_name == "Hill start":
            grade_deg = st.slider("Road grade (°)", 0.0, 15.0, 8.0, 0.5)

        gear_delay = st.slider(
            "Gear CAN delay (ms)", 0, 200, 0, 20,
            help="Simulates gateway latency causing stale gear signal"
        )
        noise_seed = st.number_input("Noise seed", value=42, step=1)

        manoeuvre_map = {
            "City drive": manoeuvre_city_drive(),
            "Reverse parking": manoeuvre_reverse_parking(),
            "Hill start": manoeuvre_hill_start(grade_deg),
            "K-turn": manoeuvre_k_turn(),
            "ABS emergency stop": manoeuvre_abs_stop(),
        }
        manoeuvre_pts = manoeuvre_map[manoeuvre_name]

        st.markdown("**Manoeuvre waypoints**")
        mp_df = pd.DataFrame([{
            "time_ms": p.time_ms,
            "throttle_%": p.throttle_pct,
            "brake_bar": p.brake_bar,
            "gear": p.gear.value,
            "grade_°": p.road_grade_deg,
        } for p in manoeuvre_pts])
        st.dataframe(mp_df, use_container_width=True, hide_index=True)

    with col_p:
        model = VehicleModel(noise_seed=int(noise_seed), gear_delay_ms=gear_delay)
        signals = model.run_manoeuvre(manoeuvre_pts)
        signal_tuples = [
            (s.timestamp_ms, s.wheel_speeds.fl, s.wheel_speeds.fr,
             s.wheel_speeds.rl, s.wheel_speeds.rr,
             s.longitudinal_acceleration, s.brake_pressure_bar,
             s.gear_position.value)
            for s in signals
        ]
        outputs = run_engine(tuple(signal_tuples), debounce_ms, agree_cycles)
        df = outputs_to_df(signal_tuples, outputs)

        render_metrics(outputs, df)
        st.plotly_chart(build_plots(df), use_container_width=True)

        if gear_delay > 0:
            mismatch_count = df["gear_mismatch"].sum()
            if mismatch_count > 0:
                st.warning(
                    f"⚠️ gear_mismatch detected in {mismatch_count} cycles "
                    f"({100*mismatch_count/len(df):.1f}%) — "
                    f"physical sensors overriding stale CAN gear signal."
                )

        st.markdown("**Diagnostic word (last cycle)**")
        render_diagnostics(outputs)
