import streamlit as st
from datetime import date
import pandas as pd
import os

# Load environment variables from .env early so config picks up PERPLEXITY_API_KEY
from dotenv import load_dotenv
load_dotenv()

# Clear SSL_CERT_FILE if it points to a missing path (prevents httpx SSL errors)
os.environ.pop("SSL_CERT_FILE", None)

# Ensure the repository root is on sys.path so `src.*` imports work under Streamlit
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import HISTORY_CSV_PATH, FORECAST_HORIZON_DAYS, OUTPUT_DIR
from src.forecast_pipeline import run_forecast
from streamlit_dashboard import render_dashboard

st.set_page_config(page_title="Steel Price Predictor", layout="wide")

# Top-level navigation: Predict or Dashboard
page = st.sidebar.radio("App", ["Predict", "Dashboard"], index=0)

if page == "Predict":
    # --- Simplified, friendly UI (single Predict flow) ---
    st.title("Steel Price Predictor — 30 Day Forecast")
    st.markdown(
        "Predict HRC steel India prices for the next 30 days. Click **Predict** to run the pipeline. "
        "Results (plot + CSV preview) appear on the right."
    )

    controls, results = st.columns([1, 2])

    with controls:
        st.header("Controls")
        today = st.date_input("Select date (defaults to today)", value=date.today())
        use_news = st.checkbox("Use news & sentiment (Perplexity API)", value=True)
        st.caption("The sentiment model runs locally and may take time at first run.")

        predict_clicked = st.button("Predict")

        if predict_clicked:
            with st.spinner("Running forecast — this may take a little while..."):
                try:
                    forecast_df, bracket_ranges, plot_path, csv_path = run_forecast(
                        data_path=HISTORY_CSV_PATH,
                        current_date=today,
                        horizon_days=FORECAST_HORIZON_DAYS,
                        use_news=use_news,
                    )
                except Exception as e:
                    st.error(f"Forecast failed: {e}")
                    raise

            # Save outputs to session state so results area can show them
            st.session_state["latest_csv"] = csv_path
            st.session_state["latest_plot"] = plot_path
            st.success("Forecast complete — see results on the right")

        st.markdown("---")
        st.subheader("Recent Forecasts")
        # show quick list of recent forecast files
        try:
            recent = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("forecast_") and f.endswith(".csv")], reverse=True)
            for f in recent[:5]:
                st.write(f)
        except Exception:
            st.write("No recent forecasts found")

    # Show results area (right column)
    with results:
        st.header("Results")
        if st.session_state.get("latest_plot") and st.session_state.get("latest_csv"):
            plot_file = st.session_state["latest_plot"]
            csv_file = st.session_state["latest_csv"]

            # Plot
            st.subheader("Forecast Plot")
            if os.path.exists(plot_file):
                st.image(plot_file, width='stretch')
            else:
                st.warning(f"Plot file not found: {plot_file}")

            # CSV preview + download
            st.subheader("Forecast CSV Preview")
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    st.dataframe(df, width='stretch')

                    with open(csv_file, "rb") as f:
                        csv_bytes = f.read()
                    st.download_button(
                        label="Download full CSV",
                        data=csv_bytes,
                        file_name=os.path.basename(csv_file),
                        mime="text/csv",
                    )

                    # Show bracket summary
                    try:
                        if 'bracket_ranges' in locals() and bracket_ranges:
                            st.markdown("---")
                            st.subheader("10-day Bracket Ranges")
                            for i, br in enumerate(bracket_ranges, start=1):
                                st.metric(label=f"Bracket {i}", value=f"{br['start_date']} → {br['end_date']}", delta=f"min={br['min_price']:.2f}, max={br['max_price']:.2f}")
                    except Exception:
                        # bracket_ranges might not be in locals if restored from session — load from CSV if needed
                        pass

                except Exception as e:
                    st.error(f"Could not read CSV: {e}")
            else:
                st.warning(f"CSV file not found: {csv_file}")
        else:
            st.info("No results yet. Click Predict to generate a forecast.")
elif page == "Dashboard":
    # Render the interactive dashboard in the same Streamlit app
    render_dashboard()


# On app start: if session state doesn't have latest results, try to restore from OUTPUT_DIR
if not st.session_state.get("latest_csv"):
    try:
        files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("forecast_") and f.endswith(".csv")]
        if files:
            files_sorted = sorted(files, reverse=True)
            latest = os.path.join(OUTPUT_DIR, files_sorted[0])
            st.session_state["latest_csv"] = latest
            # attempt to pair with plot
            date_part = files_sorted[0].split("forecast_")[1].split(".")[0]
            png_candidates = [p for p in os.listdir(OUTPUT_DIR) if date_part in p and p.endswith(".png")]
            if png_candidates:
                st.session_state["latest_plot"] = os.path.join(OUTPUT_DIR, png_candidates[0])
    except Exception:
        pass
