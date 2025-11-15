from typing import List, Dict

from perplexity import Perplexity  # from perplexityai package
from config import (
    PERPLEXITY_API_KEY,
    NEWS_QUERY_TEMPLATE,
    NEWS_MAX_RESULTS,
    NEWS_MAX_TOKENS_PER_PAGE,
)


def fetch_steel_news(
    n_results: int = NEWS_MAX_RESULTS,
    api_key: str | None = None,
) -> List[Dict]:
    """
    Fetch recent and relevant news articles related to HRC steel India prices.

    Returns a list of dicts:
      { 'title': ..., 'snippet': ..., 'url': ... }
    """
    key = api_key or PERPLEXITY_API_KEY
    if not key:
        # Gracefully handle missing API key: return empty news list
        # The rest of the pipeline treats empty news as neutral sentiment.
        print("Warning: PERPLEXITY_API_KEY is not set. Skipping news fetch and using neutral sentiment.")
        return []

    client = Perplexity(api_key=key)

    # Perplexity API enforces max_results between 1 and 20. Cap to 20 if config requests more.
    if n_results is None:
        n_results = NEWS_MAX_RESULTS
    n_results = max(1, min(int(n_results), 20))
    if n_results < NEWS_MAX_RESULTS:
        print(f"Note: limiting news fetch to {n_results} results due to API limits.")

    query = NEWS_QUERY_TEMPLATE.format(n=n_results)

    search = client.search.create(
        query=query,
        max_results=n_results,
        max_tokens_per_page=NEWS_MAX_TOKENS_PER_PAGE,
    )

    results = []
    for r in search.results:
        results.append(
            {
                "title": getattr(r, "title", ""),
                "snippet": getattr(r, "snippet", ""),
                "url": getattr(r, "url", ""),
                "date": getattr(r, "date", None),
            }
        )
    return results
