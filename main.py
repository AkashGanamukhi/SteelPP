import argparse
import os
from datetime import datetime, date

from dotenv import load_dotenv
load_dotenv()
# Some environments set SSL_CERT_FILE to a non-existent path which causes httpx to fail.
# Clear it so the stdlib/OpenSSL will use system defaults instead.
os.environ.pop("SSL_CERT_FILE", None)

from config import (
    HISTORY_CSV_PATH,
    FORECAST_HORIZON_DAYS,
    OUTPUT_DIR,
)
from src.forecast_pipeline import run_forecast


def parse_args():
    parser = argparse.ArgumentParser(description="Steel price predictor (HRC India).")
    parser.add_argument(
        "--data_path",
        type=str,
        default=HISTORY_CSV_PATH,
        help="Path to weekly price CSV.",
    )
    parser.add_argument(
        "--current_date",
        type=str,
        default=None,
        help="Current date in YYYY-MM-DD (defaults to today).",
    )
    parser.add_argument(
        "--use_news",
        type=int,
        default=1,
        help="1 to use news + sentiment, 0 for time series only.",
    )
    return parser.parse_args()


def _parse_date(s: str | None) -> date:
    if not s:
        return date.today()
    return datetime.strptime(s, "%Y-%m-%d").date()


def main():
    args = parse_args()
    current_date = _parse_date(args.current_date)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    forecast_df, bracket_ranges, plot_path, csv_path = run_forecast(
        data_path=args.data_path,
        current_date=current_date,
        horizon_days=FORECAST_HORIZON_DAYS,
        use_news=bool(args.use_news),
    )

    print("\n=== Daily Forecast (next 30 days) ===")
    print(forecast_df.head(35).to_string(index=False))  # print a bit more than 30 if any offset

    print("\n=== 10-day Bracket Ranges ===")
    for i, br in enumerate(bracket_ranges, start=1):
        print(
            f"Bracket {i}: {br['start_date']} -> {br['end_date']} | "
            f"min={br['min_price']:.2f}, max={br['max_price']:.2f}"
        )

    print(f"\nForecast CSV saved to: {csv_path}")
    print(f"Forecast plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
