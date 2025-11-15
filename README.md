<<<<<<< HEAD
# SteelPricePredictor
=======
# Steel Price Predictor (HRC India, 30-day Horizon)

This project predicts **HRC steel India** prices for the next 30 days with:

1. **Time-series model (ANN)**  
   - Trained on weekly historical prices (Aug 2021 – Nov 2025).
   - Predicts future weekly prices and interpolates to **daily** prices for the next 30 days from the current run date.
   - Outputs:
     - Daily point forecast for next 30 days.
     - 3 brackets of 10 days each with [min, max] price range.
     - PNG plot of the final forecast.

2. **News sentiment model**  
   - Uses **Perplexity Search API** to fetch top N recent news articles that can affect HRC steel India prices.
   - Uses a pretrained **Transformer sentiment model** to score each article.
   - Aggregates sentiment to adjust the time-series forecast.

3. **Fusion logic**
   - Time-series forecast = base signal.
   - Net sentiment (mean polarity in [-1, 1]) scales prices gradually across the 30-day horizon.

## Inputs

1. **Time Series CSV**  
   Path: `data/hrc_prices_weekly.csv`  
   Required columns:
   - `date` (YYYY-MM-DD)
   - `price` (float, weekly HRC India price)

2. **Perplexity Search API**
   - Sign up and create an API key from the Perplexity dashboard.
   - Install their SDK: `pip install perplexityai`.
   - Set environment variable:
     ```bash
     export PERPLEXITY_API_KEY="YOUR_API_KEY"
     ```

## Installation

```bash
cd steel-price-predictor
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
>>>>>>> 6922cbc (Update commit)
