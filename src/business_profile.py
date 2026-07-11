import json
from pathlib import Path

PROFILE_PATH = Path(__file__).resolve().parent.parent / "data" / "business_profile.json"


def save_profile(profile):
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)


def load_profile():
    if not PROFILE_PATH.exists():
        return {}
    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
