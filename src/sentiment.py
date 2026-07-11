from transformers import pipeline

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
    return _pipeline


def classify_sentiment(text):
    """Returns ("positive" | "negative", confidence)."""
    result = _get_pipeline()(text)[0]
    label = "positive" if result["label"] == "POSITIVE" else "negative"
    return label, result["score"]
