import urllib.parse

import feedparser

from utils.structured_logger import get_structured_logger

logger = get_structured_logger("sentiment_analyzer")

# Company name to Ticker Entity lookup table
ENTITY_MAP = {
    "tata motors": "TATAMOTORS",
    "tata steel": "TATASTEEL",
    "tcs": "TCS",
    "tata consultancy": "TCS",
    "reliance": "RELIANCE",
    "sbin": "SBIN",
    "state bank of india": "SBIN",
    "hdfc bank": "HDFCBANK",
    "infosys": "INFY",
    "infy": "INFY",
}

# Dynamic loading of HuggingFace FinBERT to prevent crash if not fully installed yet
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Loading local FinBERT model (ProsusAI/finbert)...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    HAS_FINBERT = True
    logger.info("FinBERT model loaded successfully.")
except Exception as e:
    logger.warning(f"Could not load FinBERT ({e}). Falling back to mock sentiment analyzer.")
    HAS_FINBERT = False


def resolve_entity(headline: str, fallback_symbol: str) -> str:
    """
    Ensures news about Tata Steel doesn't impact TCS, etc.
    Returns the resolved ticker symbol if matched, otherwise fallback_symbol.
    """
    lower_headline = headline.lower()
    for name, ticker in ENTITY_MAP.items():
        if name in lower_headline:
            return ticker
    return fallback_symbol


def fetch_local_finbert_sentiment(symbol: str) -> dict:
    """
    Real-time headline scoring using local FinBERT (ProsusAI/finbert).
    Bypasses LLM API latency (scores in 10-50ms).
    """
    clean_symbol = symbol.replace(".NS", "").replace(".BO", "")
    query = urllib.parse.quote(f"{clean_symbol} stock India OR {clean_symbol} share price")
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        feed = feedparser.parse(rss_url)
        headlines = [entry.title for entry in feed.entries[:10]]

        if not headlines:
            return {"score": 0.0, "confidence": 0.0, "key_factor": "No news found"}

        # Filter headlines based on entity resolution
        relevant_headlines = []
        for h in headlines:
            resolved = resolve_entity(h, clean_symbol)
            if resolved == clean_symbol:
                relevant_headlines.append(h)

        if not relevant_headlines:
            return {"score": 0.0, "confidence": 0.0, "key_factor": "No entity-matched news found"}

        if not HAS_FINBERT:
            # Mock sentiment fallback
            return {"score": 0.35, "confidence": 0.85, "key_factor": "Mock FinBERT (Positive cues)"}

        # Run local FinBERT inference
        inputs = tokenizer(relevant_headlines, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = F.softmax(outputs.logits, dim=-1)

        # FinBERT labels: 0 -> positive, 1 -> negative, 2 -> neutral
        # Map labels to range -1 to +1
        scores = []
        confidences = []
        for pred in predictions:
            pos, neg, neu = float(pred[0]), float(pred[1]), float(pred[2])
            # sentiment score = pos - neg
            score = pos - neg
            scores.append(score)
            confidences.append(max(pos, neg, neu))

        avg_score = sum(scores) / len(scores)
        avg_conf = sum(confidences) / len(confidences)

        return {
            "score": round(avg_score, 3),
            "confidence": round(avg_conf, 3),
            "key_factor": f"FinBERT: {len(relevant_headlines)} headlines parsed.",
        }

    except Exception as e:
        logger.error(f"Error in FinBERT sentiment for {symbol}: {e}", exc_info=True)
        return {"score": 0.0, "confidence": 0.0, "key_factor": "Error"}
