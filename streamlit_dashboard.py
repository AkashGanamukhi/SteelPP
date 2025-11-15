import streamlit as st
import pandas as pd
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

from config import OUTPUT_DIR

def find_latest_forecast_csv(directory: str) -> str | None:
    try:
        files = [f for f in os.listdir(directory) if f.startswith("forecast_") and f.endswith(".csv")]
        if not files:
            return None
        files_sorted = sorted(files, reverse=True)
        return os.path.join(directory, files_sorted[0])
    except Exception:
        return None

def render_dashboard():
    st.title("Steel Price Forecast — Interactive Dashboard")
    st.markdown("This dashboard shows the most recent forecast CSV and provides interactive charts and summary metrics.")

    # Helper: find latest forecast CSV
    csv_path = find_latest_forecast_csv(OUTPUT_DIR)

    if not csv_path or not os.path.exists(csv_path):
        st.warning("No forecast CSV found in `forecast_output`. Run a forecast first.")
        return

    st.sidebar.header("Data")
    st.sidebar.write(f"Using: `{os.path.basename(csv_path)}`")

    # Load data
    @st.cache_data
    def load_forecast(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    df = load_forecast(csv_path)

    # Top-level metrics
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.metric("Start Date", str(df["date"].min().date()))
    with col_kpi2:
        st.metric("End Date", str(df["date"].max().date()))
    with col_kpi3:
        st.metric("Base Price (avg)", f"{df['base_price'].mean():,.2f}")
    with col_kpi4:
        st.metric("Final Price (avg)", f"{df['final_price'].mean():,.2f}")

    st.markdown("---")

    # Controls
    with st.sidebar.expander("Chart Options", expanded=True):
        series = st.multiselect(
            "Select series to plot",
            options=["base_price", "final_price", "sentiment_factor"],
            default=["base_price", "final_price"],
        )
        date_range = st.date_input(
            "Date range",
            value=(df["date"].min().date(), df["date"].max().date()),
        )
        smoothing = st.slider("Rolling mean (days, 0 = no smoothing)", 0, 7, 0)

    # Filter by date
    start_date, end_date = date_range
    mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
    df_view = df.loc[mask].copy()

    # Apply smoothing if requested
    if smoothing > 0 and "base_price" in df_view.columns:
        df_view = df_view.sort_values("date")
        for s in [c for c in series if c in df_view.columns and c != "sentiment_factor"]:
            df_view[f"{s}_sm"] = df_view[s].rolling(window=smoothing, min_periods=1).mean()

    # Main chart
    st.subheader("Interactive Time Series")
    fig = go.Figure()
    color_map = {
        "base_price": "#1f77b4",
        "final_price": "#ff7f0e",
        "sentiment_factor": "#2ca02c",
    }
    for s in series:
        if s not in df_view.columns and f"{s}_sm" not in df_view.columns:
            continue
        ycol = f"{s}_sm" if f"{s}_sm" in df_view.columns else s
        mode = "lines+markers" if s != "sentiment_factor" else "lines"
        fig.add_trace(
            go.Scatter(
                x=df_view["date"],
                y=df_view[ycol],
                mode=mode,
                name=s,
                line=dict(color=color_map.get(s, None)),
                hovertemplate="%{x|%Y-%m-%d}: %{y:,.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white",
    )

    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # Impact Metrics: projected percent change and volatility
    st.subheader("Impact Metrics")
    try:
        if "final_price" in df_view.columns and len(df_view) >= 2:
            start_price = float(df_view["final_price"].iloc[0])
            end_price = float(df_view["final_price"].iloc[-1])
            proj_pct = (end_price - start_price) / start_price * 100.0

            returns = df_view["final_price"].pct_change().dropna()
            vol_pct = returns.std() * 100.0

            col_a, col_b, col_c = st.columns([1, 1, 2])
            with col_a:
                st.metric("Projected % Change", f"{proj_pct:.2f}%", delta=f"{proj_pct:.2f}%")
            with col_b:
                st.metric("Volatility (std %)", f"{vol_pct:.2f}%")
            with col_c:
                delta_fig = go.Figure()
                delta_fig.add_trace(go.Bar(x=["Start", "End"], y=[start_price, end_price], marker_color=["#1f77b4", "#ff7f0e"]))
                delta_fig.update_layout(title_text="Start vs End Price", template="plotly_white", yaxis_title="Price")
                st.plotly_chart(delta_fig, width='stretch')
        else:
            st.info("Not enough final_price data to compute impact metrics.")
    except Exception:
        st.info("Could not compute impact metrics.")

    st.markdown("---")

    # Bracket Ranges (compute here and render as cards with mini-charts)
    st.subheader("10-day Bracket Ranges")
    try:
        br_len = 10
        df_sorted = df.sort_values("date").reset_index(drop=True)
        brs = []
        for i in range(0, len(df_sorted), br_len):
            chunk = df_sorted.iloc[i : i + br_len]
            if chunk.empty:
                continue
            brs.append((i // br_len + 1, chunk))

        if brs:
            cols = st.columns(len(brs))
            for (idx, chunk), col in zip(brs, cols):
                with col:
                    sd = chunk["date"].dt.date.iloc[0]
                    ed = chunk["date"].dt.date.iloc[-1]
                    mn = chunk["final_price"].min()
                    mx = chunk["final_price"].max()
                    rng = mx - mn
                    pct = (mx - mn) / mn * 100 if mn != 0 else 0.0
                    st.markdown(f"**Bracket {idx}**")
                    st.write(f"{sd} → {ed}")
                    st.metric(label="Min", value=f"{mn:,.2f}")
                    st.metric(label="Max", value=f"{mx:,.2f}")
                    st.metric(label="Range", value=f"{rng:,.2f}", delta=f"{pct:.2f}%")
                    mini = go.Figure(go.Scatter(x=chunk["date"], y=chunk["final_price"], mode="lines", line=dict(color="#636efa")))
                    mini.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=120, template="plotly_white")
                    st.plotly_chart(mini, width='stretch')
        else:
            st.info("No bracket data available.")
    except Exception:
        st.info("Could not compute bracket ranges.")

    st.markdown("---")

    # Data table and download
    st.subheader("Forecast Data")
    st.dataframe(df_view, width='stretch')

    with open(csv_path, "rb") as f:
        csv_bytes = f.read()
    st.download_button("Download CSV", data=csv_bytes, file_name=os.path.basename(csv_path), mime="text/csv")

    st.markdown("---")
    st.caption("Dashboard generated from latest forecast in `forecast_output`.")


if __name__ == "__main__":
    # When run directly, set a page config then render
    st.set_page_config(page_title="Steel Forecast Dashboard", layout="wide")
    render_dashboard()
