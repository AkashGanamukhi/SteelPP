from __future__ import annotations

from typing import List, Dict

import numpy as np
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

from config import SENTIMENT_MODEL_NAME


class NewsSentimentModel:
    """
    Wraps a pretrained transformer sentiment model to score news snippets.

    Output:
      - label: POSITIVE / NEUTRAL / NEGATIVE (or model-specific)
      - raw_score: model probability
      - polarity: mapped value in [-1, 1]
    """

    def __init__(self, model_name: str = SENTIMENT_MODEL_NAME):
        # This will download the model the first time, so don't be surprised.
        self.pipe = pipeline(
            task="sentiment-analysis",
            model=model_name,
            truncation=True,
        )

    @staticmethod
    def _label_to_polarity(label: str, score: float) -> float:
        """Map model label+score to polarity in [-1, 1]."""
        label = label.upper()
        # CardiffNLP variant has labels like "positive", "neutral", "negative"
        if "POS" in label:
            return float(score)
        if "NEG" in label:
            return float(-score)
        # neutral or unknown gets 0
        return 0.0

    def score_texts(self, texts: List[str]) -> pd.DataFrame:
        if not texts:
            return pd.DataFrame(columns=["text", "label", "raw_score", "polarity"])

        results = []
        for t in tqdm(texts, desc="Scoring sentiment", leave=False):
            out = self.pipe(t[:512])[0]  # truncate for safety
            label = out["label"]
            score = float(out["score"])
            polarity = self._label_to_polarity(label, score)
            results.append(
                {
                    "text": t,
                    "label": label,
                    "raw_score": score,
                    "polarity": polarity,
                }
            )

        return pd.DataFrame(results)

    def score_news_items(self, news_items: List[Dict]) -> pd.DataFrame:
        """
        news_items: list of dict {title, snippet, url, date}
        """
        texts = []
        for item in news_items:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            combined = f"{title}. {snippet}".strip()
            if not combined:
                continue
            texts.append(combined)

        df_scores = self.score_texts(texts)
        return df_scores
