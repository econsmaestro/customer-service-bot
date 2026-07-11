import json
from datetime import datetime, timezone
from pathlib import Path

FEEDBACK_LOG = Path(__file__).resolve().parent.parent / "data" / "feedback_log.jsonl"


def save_feedback(message, confidence, sentiment, sentiment_confidence):
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": message,
        "confidence": confidence,
        "sentiment": sentiment,
        "sentiment_confidence": sentiment_confidence,
    }
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_feedback():
    if not FEEDBACK_LOG.exists():
        return []
    entries = []
    with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
