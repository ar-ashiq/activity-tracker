"""Rule-based plain-English queries over consolidated activity predictions."""

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


ACTIVITY_WORDS = {
    "LYING_DOWN": ["lying down", "lying", "in bed", "asleep", "sleeping"],
    "SITTING": ["sitting", "sit", "sat"],
    "STANDING": ["standing", "stand"],
    "WALKING": ["walking", "walk"],
    "RUNNING": ["running", "run", "jogging", "jog"],
    "BICYCLING": ["biking", "bicycling", "cycling", "bike"],
}


def _label(activity: str) -> str:
    return activity.replace("_", " ").lower()


def _canonical_activity(activity: str) -> str:
    normalized = activity.strip().upper().replace(" ", "_")
    return normalized


def _format_timestamp(timestamp: float) -> str:
    return f"{timestamp:g}"


def _find_activity_in_text(text: str) -> Optional[str]:
    text = text.lower()
    for activity, words in ACTIVITY_WORDS.items():
        if any(re.search(rf"\b{re.escape(word)}\b", text) for word in words):
            return activity
    return None


def _find_time_in_text(text: str) -> Optional[float]:
    numeric_match = re.search(r"\b(?:at|around)\s+(\d+(?:\.\d+)?)\b", text.lower())
    if numeric_match:
        return float(numeric_match.group(1))

    clock_match = re.search(r"(\d{1,2}):(\d{2})\s*(am|pm)?", text.lower())
    if not clock_match:
        return None

    hour = int(clock_match.group(1))
    minute = int(clock_match.group(2))
    meridiem = clock_match.group(3)
    if meridiem == "pm" and hour != 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0

    return float(hour * 3600 + minute * 60)


def _find_time_range(text: str) -> Optional[tuple[float, float]]:
    """Find ranges expressed as 'from A to B' or 'between A and B'."""
    range_match = re.search(
        r"(?:from|between)\s+(\d+(?:\.\d+)?)\s*(?:to|and)\s+"
        r"(\d+(?:\.\d+)?)",
        text.lower(),
    )
    if not range_match:
        return None

    start = float(range_match.group(1))
    end = float(range_match.group(2))
    return (start, end) if start <= end else (end, start)


def _filter_timeline(
    timeline: List[Dict[str, Any]],
    time_range: Optional[tuple[float, float]],
) -> List[Dict[str, Any]]:
    if time_range is None:
        return timeline

    range_start, range_end = time_range
    scoped = []
    for segment in timeline:
        if (
            segment["end_timestamp"] < range_start
            or segment["start_timestamp"] > range_end
        ):
            continue

        clipped = dict(segment)
        clipped["start_timestamp"] = max(segment["start_timestamp"], range_start)
        clipped["end_timestamp"] = min(segment["end_timestamp"], range_end)
        clipped["start_ms"] = clipped["start_timestamp"] * 1000
        clipped["end_ms"] = clipped["end_timestamp"] * 1000
        clipped["start_time"] = _format_clock(clipped["start_timestamp"])
        clipped["end_time"] = _format_clock(clipped["end_timestamp"])
        clipped["duration_s"] = (
            clipped["end_timestamp"] - clipped["start_timestamp"]
        )
        scoped.append(clipped)

    return scoped


def _activity_totals(timeline: List[Dict[str, Any]]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for segment in timeline:
        activity = segment["activity"]
        totals[activity] = totals.get(activity, 0.0) + segment["duration_s"]
    return totals


def _format_clock(timestamp: float) -> str:
    if timestamp < 86400:
        return _format_timestamp(timestamp)

    value = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return value.strftime("%H:%M:%S")


def build_timeline(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert consolidated JSON records into query-engine timeline records."""
    timeline = []
    for prediction in predictions:
        start = float(prediction["start_timestamp"])
        end = float(prediction["end_timestamp"])
        if end < start:
            continue

        activity = _canonical_activity(str(prediction["activity"]))
        timeline.append(
            {
                "activity": activity,
                "start_timestamp": start,
                "end_timestamp": end,
                "start_ms": start * 1000,
                "end_ms": end * 1000,
                "start_time": _format_clock(start),
                "end_time": _format_clock(end),
                "duration_s": end - start,
            }
        )

    return sorted(timeline, key=lambda segment: segment["start_timestamp"])


def answer(question: str, timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Answer a supported plain-English question with the records used."""
    query = question.strip()
    query_lower = query.lower()
    activity = _find_activity_in_text(query_lower)
    timestamp = _find_time_in_text(query_lower)
    time_range = _find_time_range(query_lower)

    if time_range is not None:
        timeline = _filter_timeline(timeline, time_range)

    if timestamp is not None and ("what" in query_lower or "doing" in query_lower):
        match = next(
            (
                segment
                for segment in timeline
                if segment["start_timestamp"] <= timestamp <= segment["end_timestamp"]
            ),
            None,
        )
        if match:
            return {
                "question": query,
                "answer": (
                    f"You were {_label(match['activity'])} from "
                    f"{match['start_time']} to {match['end_time']}."
                ),
                "evidence": [match],
            }
        return {
            "question": query,
            "answer": "No classified activity covers that time.",
            "evidence": [],
        }

    if activity and any(
        phrase in query_lower for phrase in ("how long", "how much time", "total time")
    ):
        matches = [segment for segment in timeline if segment["activity"] == activity]
        total_seconds = sum(segment["duration_s"] for segment in matches)
        time_scope = "during this time frame" if time_range is not None else "today"
        return {
            "question": query,
            "answer": (
                f"You spent about {round(total_seconds / 60, 1)} minutes "
                f"{_label(activity)} {time_scope}, across {len(matches)} segment(s)."
            ),
            "evidence": matches,
        }

    if activity and "when" in query_lower:
        matches = [segment for segment in timeline if segment["activity"] == activity]
        if not matches:
            return {
                "question": query,
                "answer": f"No {_label(activity)} segments were found.",
                "evidence": [],
            }
        spans = ", ".join(
            f"{segment['start_time']}-{segment['end_time']}" for segment in matches
        )
        return {
            "question": query,
            "answer": f"You were {_label(activity)} during: {spans}.",
            "evidence": matches,
        }

    if activity and ("how many times" in query_lower or "how many segments" in query_lower):
        matches = [segment for segment in timeline if segment["activity"] == activity]
        return {
            "question": query,
            "answer": f"{len(matches)} {_label(activity)} segment(s) were detected.",
            "evidence": matches,
        }

    overview_question = (
        "most" in query_lower
        or (
            "what" in query_lower
            and any(word in query_lower for word in ("doing", "activity", "user"))
        )
    )
    if overview_question:
        totals = _activity_totals(timeline)

        if not totals:
            return {"question": query, "answer": "No activity was recorded.", "evidence": []}

        top_activity = max(totals, key=totals.get)
        matches = [segment for segment in timeline if segment["activity"] == top_activity]
        return {
            "question": query,
            "answer": (
                f"The user was {_label(top_activity)} the most during this time frame, "
                f"for about {round(totals[top_activity] / 60, 1)} minutes."
            ),
            "evidence": matches,
        }

    if not timeline:
        return {
            "question": query,
            "answer": "No timeline is available yet - run the prediction pipeline first.",
            "evidence": [],
        }

    return {
        "question": query,
        "answer": (
            "I could not parse that question. Try: "
            '"what was I doing at 600?", "how long did I spend walking?", '
            '"when did I go running?", or "what did I do most today?".'
        ),
        "evidence": timeline[:3],
    }
