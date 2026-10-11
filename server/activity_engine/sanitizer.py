import json
from typing import Any, Optional

from server.activity_engine.config import VALID_INTENTS

VALID_ACTIVITIES = {
    "WALKING",
    "RUNNING",
    "SITTING",
    "STANDING",
    "LYING_DOWN",
    "BICYCLING",
}


def sanitize(raw: str) -> dict[str, Any]:
    try:
        decoder = json.JSONDecoder()
        start = raw.find("{")
        if start < 0:
            return fallback()
        obj, _ = decoder.raw_decode(raw[start:])

        if obj.get("intent") not in VALID_INTENTS:
            return fallback()

        intent = obj["intent"]
        summary_type = obj.get("summary_type")
        if intent == "activity_summary" and summary_type not in {"most", "least"}:
            return fallback()
        if intent != "activity_summary":
            summary_type = None

        activity = obj.get("activity")
        if activity is not None:
            activity = str(activity).strip().upper().replace(" ", "_")
            if activity not in VALID_ACTIVITIES:
                return fallback()

        compare_activity = obj.get("compare_activity")
        if compare_activity is not None:
            compare_activity = str(compare_activity).strip().upper().replace(" ", "_")
            if compare_activity not in VALID_ACTIVITIES:
                return fallback()

        raw_time = obj.get("time")
        if not isinstance(raw_time, dict):
            return fallback()

        time = {
            key: _number_or_none(raw_time.get(key))
            for key in ("at", "from", "to")
        }

        if intent == "activity_at_time" and time["at"] is None:
            return fallback()
        if intent in {"activity_duration", "activity_time_range", "activity_count", "activity_presence"} and activity is None:
            return fallback()
        if intent == "activity_compare" and (
            activity is None or compare_activity is None or activity == compare_activity
        ):
            return fallback()
        if time["from"] is not None and time["to"] is not None and time["from"] > time["to"]:
            time["from"], time["to"] = time["to"], time["from"]

        return {
            "intent": intent,
            "activity": activity,
            "compare_activity": compare_activity,
            "summary_type": summary_type,
            "time": time,
        }

    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback()


def _number_or_none(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fallback():
    return {
        "intent": "out_of_scope",
        "activity": None,
        "compare_activity": None,
        "summary_type": None,
        "time": {"at": None, "from": None, "to": None}
    }