import json
import re
from server.activity_engine.config import VALID_INTENTS


def sanitize(raw: str):

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return fallback()

    try:
        obj = json.loads(match.group())

        if obj.get("intent") not in VALID_INTENTS:
            return fallback()

        if "time" not in obj:
            obj["time"] = {}

        return obj

    except:
        return fallback()


def fallback():
    return {
        "intent": "activity_summary",
        "activity": None,
        "time": {"at": None, "from": None, "to": None}
    }