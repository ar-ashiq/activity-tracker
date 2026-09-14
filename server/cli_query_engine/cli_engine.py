import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# CONFIG
# ============================================================

ACTIVITY_WORDS = {
    "LYING_DOWN": ["lying down", "lying", "in bed", "asleep", "sleeping"],
    "SITTING": ["sitting", "sit", "sat"],
    "STANDING": ["standing", "stand"],
    "WALKING": ["walking", "walk"],
    "RUNNING": ["running", "run", "jogging", "jog"],
    "BICYCLING": ["biking", "bicycling", "cycling", "bike"],
}


# ============================================================
# UTILITIES
# ============================================================

def _canonical_activity(activity: str) -> str:
    return activity.strip().upper().replace(" ", "_")


def _label(activity: str) -> str:
    return activity.replace("_", " ").lower()


def _format_clock(timestamp: float) -> str:
    if timestamp < 86400:
        return f"{timestamp:g}"
    value = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return value.strftime("%H:%M:%S")


# ============================================================
# TEXT PARSING
# ============================================================

def _find_activity_in_text(text: str) -> Optional[str]:
    text = text.lower()
    for activity, words in ACTIVITY_WORDS.items():
        if any(re.search(rf"\b{re.escape(word)}\b", text) for word in words):
            return activity
    return None


def _find_time_in_text(text: str) -> Optional[float]:
    match = re.search(r"\b(?:at|around)\s+(\d+(?:\.\d+)?)", text.lower())
    if match:
        return float(match.group(1))

    clock = re.search(r"(\d{1,2}):(\d{2})\s*(am|pm)?", text.lower())
    if not clock:
        return None

    hour = int(clock.group(1))
    minute = int(clock.group(2))
    meridiem = clock.group(3)

    if meridiem == "pm" and hour != 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0

    return float(hour * 3600 + minute * 60)


def _find_time_range(text: str) -> Optional[Tuple[float, float]]:
    match = re.search(
        r"(?:from|between)\s+(\d+(?:\.\d+)?)\s*(?:to|and)\s+(\d+(?:\.\d+)?)",
        text.lower(),
    )
    if not match:
        return None

    a = float(match.group(1))
    b = float(match.group(2))
    return (min(a, b), max(a, b))


# ============================================================
# TIMELINE BUILDING
# ============================================================

def build_timeline(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    timeline = []

    for p in predictions:
        start = float(p["start_timestamp"])
        end = float(p["end_timestamp"])

        if end < start:
            continue

        timeline.append(
            {
                "activity": _canonical_activity(str(p["activity"])),
                "start_timestamp": start,
                "end_timestamp": end,
                "start_time": _format_clock(start),
                "end_time": _format_clock(end),
                "duration_s": end - start,
            }
        )

    return sorted(timeline, key=lambda x: x["start_timestamp"])


def _filter_timeline(
    timeline: List[Dict[str, Any]],
    time_range: Optional[Tuple[float, float]],
) -> List[Dict[str, Any]]:

    if not time_range:
        return timeline

    start_r, end_r = time_range
    filtered = []

    for seg in timeline:
        if seg["end_timestamp"] < start_r or seg["start_timestamp"] > end_r:
            continue

        clipped = dict(seg)
        clipped["start_timestamp"] = max(seg["start_timestamp"], start_r)
        clipped["end_timestamp"] = min(seg["end_timestamp"], end_r)
        clipped["duration_s"] = clipped["end_timestamp"] - clipped["start_timestamp"]

        filtered.append(clipped)

    return filtered


def _activity_totals(timeline: List[Dict[str, Any]]) -> Dict[str, float]:
    totals = {}

    for seg in timeline:
        totals[seg["activity"]] = totals.get(seg["activity"], 0) + seg["duration_s"]

    return totals


# ============================================================
# CORE QUERY ENGINE
# ============================================================

def answer(question: str, timeline: List[Dict[str, Any]]) -> Dict[str, Any]:

    q = question.lower()

    activity = _find_activity_in_text(q)
    timestamp = _find_time_in_text(q)
    time_range = _find_time_range(q)

    if time_range:
        timeline = _filter_timeline(timeline, time_range)

    # --------------------------------------------------------
    # 1. What was I doing at X?
    # --------------------------------------------------------
    if timestamp is not None and ("what" in q or "doing" in q):

        match = next(
            (
                s for s in timeline
                if s["start_timestamp"] <= timestamp <= s["end_timestamp"]
            ),
            None,
        )

        if match:
            return {
                "question": question,
                "answer": f"You were {_label(match['activity'])} at {timestamp:g}s.",
                "evidence": [match],
            }

        return {
            "question": question,
            "answer": "No activity found for that time.",
            "evidence": [],
        }

    # --------------------------------------------------------
    # 2. How long did I do X?
    # --------------------------------------------------------
    if activity and "how long" in q:
        matches = [s for s in timeline if s["activity"] == activity]
        total = sum(m["duration_s"] for m in matches)

        return {
            "question": question,
            "answer": f"You spent ~{round(total/60, 2)} minutes {_label(activity)}.",
            "evidence": matches,
        }

    # --------------------------------------------------------
    # 3. When was I doing X?
    # --------------------------------------------------------
    if activity and "when" in q:
        matches = [s for s in timeline if s["activity"] == activity]

        if not matches:
            return {
                "question": question,
                "answer": f"No {_label(activity)} detected.",
                "evidence": [],
            }

        spans = ", ".join(
            f"{m['start_time']}–{m['end_time']}" for m in matches
        )

        return {
            "question": question,
            "answer": f"You were {_label(activity)} during: {spans}",
            "evidence": matches,
        }

    # --------------------------------------------------------
    # 4. How many times X?
    # --------------------------------------------------------
    if activity and "how many" in q:
        matches = [s for s in timeline if s["activity"] == activity]

        return {
            "question": question,
            "answer": f"{len(matches)} segments of {_label(activity)} detected.",
            "evidence": matches,
        }

    # --------------------------------------------------------
    # 5. Summary / most activity
    # --------------------------------------------------------
    if "most" in q or "summary" in q or "overall" in q:

        totals = _activity_totals(timeline)

        if not totals:
            return {
                "question": question,
                "answer": "No data available.",
                "evidence": [],
            }

        top = max(totals, key=totals.get)

        return {
            "question": question,
            "answer": f"Most activity: {_label(top)} (~{round(totals[top]/60,2)} min)",
            "evidence": [s for s in timeline if s["activity"] == top],
        }

    # --------------------------------------------------------
    # fallback
    # --------------------------------------------------------
    return {
        "question": question,
        "answer": "Query not understood. Try: 'what was I doing at 600?' or 'how long did I walk?'",
        "evidence": [],
    }


# ============================================================
# JSON LOADER
# ============================================================

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def load_timeline(path: str):
    return build_timeline(load_json(path))


# ============================================================
# CLI
# ============================================================

def run_cli(json_path: str):
    timeline = load_timeline(json_path)

    print("\nActivity Query Engine Ready")
    print("Type 'exit' to quit\n")

    while True:
        q = input("Query: ")
        if q.strip().lower() == "exit":
            break

        res = answer(q, timeline)

        print("\nAnswer:", res["answer"])
        print("Evidence:", len(res["evidence"]))
        print()


# ============================================================
# OPTIONAL FASTAPI (UNCOMMENT IF NEEDED)
# ============================================================

"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
timeline = None

class Query(BaseModel):
    question: str

@app.on_event("startup")
def startup():
    global timeline
    timeline = load_timeline("predictions.json")

@app.post("/query")
def query(q: Query):
    return answer(q.question, timeline)
"""


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python activity_query_engine.py <json_file>")
    else:
        run_cli(sys.argv[1])