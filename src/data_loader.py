import pandas as pd
from config import DATE_COLUMN, PRICE_COLUMN


def _find_column(df: pd.DataFrame, expected: str) -> str | None:
    # Exact match
    if expected in df.columns:
        return expected

    # Case-insensitive exact
    lower_map = {c.lower(): c for c in df.columns}
    if expected.lower() in lower_map:
        return lower_map[expected.lower()]

    # Heuristic matches for common alternatives
    for c in df.columns:
        lc = c.lower()
        if expected.lower() == "date" and ("date" in lc or "day" in lc):
            return c
        if expected.lower() == "price" and ("price" in lc or "inr" in lc or "value" in lc):
            return c

    return None


def load_weekly_price_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    date_col = _find_column(df, DATE_COLUMN)
    if date_col is None:
        raise ValueError(f"CSV must contain '{DATE_COLUMN}' column (or a common alternative like 'Date').")

    price_col = _find_column(df, PRICE_COLUMN)
    if price_col is None:
        raise ValueError(f"CSV must contain '{PRICE_COLUMN}' column (or a common alternative like 'Price' or 'Price_INR').")

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    if df[date_col].isna().all():
        raise ValueError(f"Unable to parse dates from column '{date_col}'. Expected format like YYYY-MM-DD or DD-MM-YYYY.")

    df = df.sort_values(date_col).reset_index(drop=True)

    # Force weekly frequency by dropping duplicates and resampling if needed
    df = df.drop_duplicates(subset=[date_col])

    # Return with expected column names for downstream code
    df = df[[date_col, price_col]]
    df = df.rename(columns={date_col: DATE_COLUMN, price_col: PRICE_COLUMN})
    return df
