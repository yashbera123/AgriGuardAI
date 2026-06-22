"""Analytics dashboard helpers for AgriGuard AI."""

import pandas as pd
import plotly.express as px
import streamlit as st

from prediction_logger import HISTORY_COLUMNS


def prepare_history(history_df):
    """Normalize history data before metrics and charts are calculated."""
    if history_df is None or history_df.empty:
        return pd.DataFrame(columns=HISTORY_COLUMNS + ["Timestamp Parsed"])

    history = history_df.copy()

    for column in HISTORY_COLUMNS:
        if column not in history.columns:
            history[column] = pd.NA

    history["Confidence"] = pd.to_numeric(
        history["Confidence"],
        errors="coerce",
    )
    history["Sustainability Score"] = pd.to_numeric(
        history["Sustainability Score"],
        errors="coerce",
    )
    history["Timestamp Parsed"] = pd.to_datetime(
        history["Timestamp"],
        errors="coerce",
    )

    return history


def summarize_history(history_df):
    """Return headline analytics for prediction history."""
    history = prepare_history(history_df)

    if history.empty:
        return {
            "total_predictions": 0,
            "most_common_disease": "No data",
            "average_confidence": 0.0,
            "average_sustainability": 0.0,
        }

    disease_counts = history["Disease Name"].dropna().value_counts()
    average_confidence = history["Confidence"].mean(skipna=True)
    average_sustainability = history["Sustainability Score"].mean(skipna=True)

    return {
        "total_predictions": int(len(history)),
        "most_common_disease": (
            disease_counts.idxmax() if not disease_counts.empty else "No data"
        ),
        "average_confidence": (
            float(average_confidence) if pd.notna(average_confidence) else 0.0
        ),
        "average_sustainability": (
            float(average_sustainability) if pd.notna(average_sustainability) else 0.0
        ),
    }


def render_analytics_dashboard(history_df):
    """Render the Farm Analytics Dashboard in Streamlit."""
    st.write("## Farm Analytics Dashboard")

    history = prepare_history(history_df)

    if history.empty:
        st.info("No prediction history yet. Run a prediction to start the dashboard.")
        return

    summary = summarize_history(history)

    # Dashboard headline metrics.
    metric_cols = st.columns(4)
    metric_cols[0].metric("Total Predictions", summary["total_predictions"])
    metric_cols[1].metric("Most Common Disease", summary["most_common_disease"])
    metric_cols[2].metric(
        "Average Confidence",
        f"{summary['average_confidence']:.2f}%",
    )
    metric_cols[3].metric(
        "Average Sustainability",
        f"{summary['average_sustainability']:.1f}/100",
    )

    disease_counts = (
        history["Disease Name"]
        .fillna("Unknown")
        .value_counts()
        .reset_index()
    )
    disease_counts.columns = ["Disease Name", "Predictions"]

    chart_col1, chart_col2 = st.columns(2)

    # Disease distribution pie chart.
    with chart_col1:
        pie_chart = px.pie(
            disease_counts,
            names="Disease Name",
            values="Predictions",
            title="Disease Distribution",
            hole=0.35,
        )
        st.plotly_chart(pie_chart, use_container_width=True)

    # Disease frequency bar chart.
    with chart_col2:
        bar_chart = px.bar(
            disease_counts,
            x="Disease Name",
            y="Predictions",
            title="Disease Frequency",
            text="Predictions",
        )
        bar_chart.update_layout(xaxis_title="", yaxis_title="Predictions")
        st.plotly_chart(bar_chart, use_container_width=True)

    timeline = history.dropna(subset=["Timestamp Parsed"]).copy()

    if not timeline.empty:
        timeline["Date"] = timeline["Timestamp Parsed"].dt.date
        timeline_counts = (
            timeline.groupby("Date")
            .size()
            .reset_index(name="Predictions")
            .sort_values("Date")
        )

        # Prediction timeline chart.
        timeline_chart = px.line(
            timeline_counts,
            x="Date",
            y="Predictions",
            markers=True,
            title="Prediction Timeline",
        )
        timeline_chart.update_layout(
            xaxis_title="Date",
            yaxis_title="Predictions",
        )
        st.plotly_chart(timeline_chart, use_container_width=True)
    else:
        st.warning("Prediction timeline is unavailable because timestamps could not be read.")

    with st.expander("View Prediction History"):
        st.dataframe(history[HISTORY_COLUMNS], use_container_width=True)
