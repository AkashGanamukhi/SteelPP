import os
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd

from config import OUTPUT_DIR, PLOT_WIDTH, PLOT_HEIGHT


def plot_forecast(
    df: pd.DataFrame,
    current_date: date,
    output_filename: str | None = None,
) -> str:
    """
    Plot final forecast for the next 30 days.
    Uses 'date' and 'final_price' columns.
    """
    if output_filename is None:
        output_filename = f"forecast_plot_{current_date.strftime('%Y%m%d')}.png"

    output_path = os.path.join(OUTPUT_DIR, output_filename)

    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    plt.plot(df["date"], df["final_price"], marker="o", label="Forecast (final)")
    if "base_price" in df.columns:
        plt.plot(df["date"], df["base_price"], linestyle="--", label="Base (time series)")
    plt.axvline(x=pd.to_datetime(current_date), linestyle=":", label="Run date")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("HRC Steel India Price Forecast (Next 30 Days)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path
