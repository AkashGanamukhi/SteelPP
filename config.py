import os
from datetime import timedelta

# ---------- Data config ----------
HISTORY_CSV_PATH = "data/hrc_prices_weekly.csv"

DATE_COLUMN = "date"
PRICE_COLUMN = "price"

# ---------- Time-series model config ----------
N_LAGS = 4  # how many past weeks to use as input
TEST_SIZE_FRACTION = 0.1
RANDOM_STATE = 42

ANN_HIDDEN_LAYER_SIZES = (64, 32)
ANN_MAX_ITER = 500

# How far ahead to predict (days from current date)
FORECAST_HORIZON_DAYS = 30

# ---------- Sentiment / News config ----------
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")

NEWS_QUERY_TEMPLATE = (
    "Give {n} recent and most relevant news article briefs that could "
    "potentially increase or decrease HRC steel India prices. "
    "Return concise bullet-style summaries."
)

NEWS_MAX_RESULTS = 25  # between 20 and 30 as requested
NEWS_MAX_TOKENS_PER_PAGE = 512

# Pretrained sentiment model from HuggingFace
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Scale factor for sentiment impact:
# final_price = base_price * (1 + SENTIMENT_ALPHA * avg_sentiment * t/T)
# where avg_sentiment in [-1, 1], t is day index, T is horizon.
SENTIMENT_ALPHA = 0.08  # ~±8% drift at day 30 for extreme sentiment

# ---------- Output paths ----------
OUTPUT_DIR = "forecast_output"
PLOT_WIDTH = 10
PLOT_HEIGHT = 5

# ---------- Brackets ----------
BRACKET_SIZE_DAYS = 10  # 3 brackets of 10 days each for 30 days

# ---------- Utility ----------
ONE_DAY = timedelta(days=1)
