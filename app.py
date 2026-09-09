"""Interactive wearable sensor data-quality dashboard."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

STATUS_COLOURS = {
    "PASS": "#1D9A6C",
    "REVIEW": "#E09F3E",
    "FAIL": "#D1495B",
}

st.set_page_config(
    page_title="Wearable Sensor Review",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1450px;
        }

        [data-testid="stMetric"] {
            background: #F7F9FC;
            border: 1px solid #E4E9F0;
            border-radius: 10px;
            padding: 14px 18px;
        }

        .status-card {
            padding: 14px 18px;
            border-radius: 10px;
            color: white;
            font-weight: 600;
            margin-bottom: 15px;
        }

        h1, h2, h3 {
            color: #17324D;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_outputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load validated outputs produced by the processing pipeline."""
    required_files = {
        "timeseries": PROCESSED_DIR / "stress_timeseries_1s.csv",
        "summary": PROCESSED_DIR / "session_summary.csv",
        "quality": PROCESSED_DIR / "quality_checks.csv",
    }

    missing_files = [
        path.name
        for path in required_files.values()
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Missing pipeline outputs: {', '.join(missing_files)}"
        )

    timeseries = pd.read_csv(
        required_files["timeseries"],
        parse_dates=["timestamp"],
    )
    summary = pd.read_csv(
        required_files["summary"],
        parse_dates=["start_time", "end_time"],
    )
    quality = pd.read_csv(required_files["quality"])

    if quality["detected"].dtype != bool:
        quality["detected"] = (
            quality["detected"]
            .astype(str)
            .str.lower()
            .eq("true")
        )

    return timeseries, summary, quality


def build_signal_figure(
    session_data: pd.DataFrame,
) -> go.Figure:
    """Build aligned HR, EDA, and movement timelines."""
    session_data = session_data.sort_values("timestamp").copy()
    start_time = session_data["timestamp"].min()

    session_data["elapsed_minutes"] = (
        session_data["timestamp"] - start_time
    ).dt.total_seconds() / 60

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=(
            "Heart Rate",
            "Electrodermal Activity",
            "Movement Variability",
        ),
    )

    figure.add_trace(
        go.Scatter(
            x=session_data["elapsed_minutes"],
            y=session_data["heart_rate_bpm"],
            mode="lines",
            name="Heart rate",
            line={"color": "#D1495B", "width": 1.5},
            hovertemplate=(
                "Time: %{x:.2f} min"
                "<br>HR: %{y:.1f} BPM<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=session_data["elapsed_minutes"],
            y=session_data["eda_microsiemens"],
            mode="lines",
            name="EDA",
            line={"color": "#3366CC", "width": 1.5},
            hovertemplate=(
                "Time: %{x:.2f} min"
                "<br>EDA: %{y:.3f} µS<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=session_data["elapsed_minutes"],
            y=session_data["movement_variability_g"],
            mode="lines",
            name="Movement",
            line={"color": "#1D9A6C", "width": 1.3},
            hovertemplate=(
                "Time: %{x:.2f} min"
                "<br>Variability: %{y:.3f} g<extra></extra>"
            ),
        ),
        row=3,
        col=1,
    )

    figure.update_yaxes(title_text="BPM", row=1, col=1)
    figure.update_yaxes(title_text="µS", row=2, col=1)
    figure.update_yaxes(title_text="g", row=3, col=1)
    figure.update_xaxes(
        title_text="Elapsed time (minutes)",
        row=3,
        col=1,
    )

    figure.update_layout(
        height=720,
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
        showlegend=False,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    figure.update_xaxes(
        showgrid=True,
        gridcolor="#EDF1F5",
        rangeslider_visible=False,
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor="#EDF1F5",
        zeroline=False,
    )

    return figure


try:
    timeseries, summary, quality = load_outputs()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()


st.title("Wearable Sensor Data Review")
st.caption(
    "Automated technical quality screening and synchronized "
    "visualisation of stress-session wearable data."
)

total_sessions = len(summary)
pass_sessions = int(summary["quality_status"].eq("PASS").sum())
review_sessions = int(summary["quality_status"].eq("REVIEW").sum())
median_coverage = summary["complete_seconds_percent"].median()

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric("Sessions received", total_sessions)
metric_2.metric("Passed screening", pass_sessions)
metric_3.metric("Require review", review_sessions)
metric_4.metric("Median coverage", f"{median_coverage:.1f}%")

overview_tab, explorer_tab = st.tabs(
    ["Review Queue", "Session Explorer"]
)

with overview_tab:
    st.subheader("Session Review Queue")
    st.caption(
        "Sessions are prioritised using automated technical checks. "
        "A REVIEW status requires human assessment, not automatic exclusion."
    )

    status_filter = st.multiselect(
        "Quality status",
        options=["REVIEW", "FAIL", "PASS"],
        default=["REVIEW", "FAIL", "PASS"],
    )

    queue = summary[
        summary["quality_status"].isin(status_filter)
    ].copy()

    status_order = {
        "FAIL": 0,
        "REVIEW": 1,
        "PASS": 2,
    }
    queue["status_order"] = queue["quality_status"].map(status_order)
    queue = queue.sort_values(
        ["status_order", "session_id"]
    )

    display_columns = [
        "session_id",
        "protocol_version",
        "duration_minutes",
        "signals_received",
        "complete_seconds_percent",
        "detected_issue_count",
        "quality_status",
        "detected_issues",
    ]

    st.dataframe(
        queue[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "session_id": "Session",
            "protocol_version": "Protocol",
            "duration_minutes": st.column_config.NumberColumn(
                "Duration",
                format="%.1f min",
            ),
            "signals_received": "Signals",
            "complete_seconds_percent":
                st.column_config.ProgressColumn(
                    "Complete coverage",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                ),
            "detected_issue_count": "Issues",
            "quality_status": "Status",
            "detected_issues": "Automated findings",
        },
    )

with explorer_tab:
    st.subheader("Participant Signal Explorer")

    filter_col_1, filter_col_2 = st.columns(2)

    with filter_col_1:
        selected_protocol = st.selectbox(
            "Protocol version",
            sorted(summary["protocol_version"].unique()),
        )

    available_sessions = summary.loc[
        summary["protocol_version"] == selected_protocol,
        "session_id",
    ].sort_values()

    with filter_col_2:
        selected_session = st.selectbox(
            "Session",
            available_sessions,
        )

    selected_summary = summary[
        summary["session_id"] == selected_session
    ].iloc[0]

    selected_data = timeseries[
        timeseries["session_id"] == selected_session
    ].copy()

    selected_checks = quality[
        (quality["session_id"] == selected_session)
        & quality["detected"]
    ].copy()

    status = selected_summary["quality_status"]
    status_colour = STATUS_COLOURS.get(status, "#61758A")

    st.markdown(
        f"""
        <div class="status-card"
             style="background-color: {status_colour};">
            {selected_session}: {status}
        </div>
        """,
        unsafe_allow_html=True,
    )

    detail_1, detail_2, detail_3, detail_4 = st.columns(4)

    detail_1.metric(
        "Duration",
        f"{selected_summary['duration_minutes']:.1f} min",
    )
    detail_2.metric(
        "Signals received",
        selected_summary["signals_received"],
    )
    detail_3.metric(
        "Complete coverage",
        f"{selected_summary['complete_seconds_percent']:.1f}%",
    )
    detail_4.metric(
        "Detected issues",
        int(selected_summary["detected_issue_count"]),
    )

    st.plotly_chart(
        build_signal_figure(selected_data),
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
        },
    )

    st.subheader("Automated Quality Evidence")

    if selected_checks.empty:
        st.success(
            "No technical quality rules were triggered for this session."
        )
    else:
        st.warning(
            "This session requires review. A triggered rule does not "
            "by itself prove that the sensor failed."
        )

        st.dataframe(
            selected_checks[
                [
                    "signal",
                    "check_name",
                    "severity",
                    "metric_value",
                    "threshold",
                    "message",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

st.divider()
st.caption(
    "Technical screening prototype using de-identified public "
    "PhysioNet wearable data. Not intended for clinical diagnosis."
)