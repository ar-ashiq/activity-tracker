import json
import re
import requests
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

VALID_INTENTS = {
    "activity_at_time",
    "activity_duration",
    "activity_time_range",
    "activity_count",
    "activity_summary"
}

def validate_intent(intent):
    if intent not in VALID_INTENTS:
        return "activity_summary"
    return intent


# ============================================================
# PHI-3 INTENT PARSER
# ============================================================

def call_phi3(question: str) -> Dict[str, Any]:
    prompt = f"""
You are a STRICT JSON FUNCTION.

RULES:
- Output ONLY valid JSON
- NO explanations
- NO markdown
- NO extra text
- If you fail, output: {{"intent":"activity_summary","activity":null,"time":{{}}}}

SCHEMA:
{{
  "intent": "activity_at_time | activity_duration | activity_time_range | activity_count | activity_summary",
  "activity": "WALKING | RUNNING | SITTING | STANDING | LYING_DOWN | BICYCLING | null",
  "time": {{
    "at": number or null,
    "from": number or null,
    "to": number or null
  }}
}}

Question:
{question}
"""

    print("\n🔵 Sending to Phi-3...")

    res = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "phi3",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.1
        }
    }
)

    raw = res.json()["response"]

    print("\n🟢 Raw Phi-3 response:")
    print(raw)

    try:
        return safe_parse_phi3(raw)
    except:
        return {
            "intent": "activity_summary",
            "activity": None,
            "time": {}
        }


# ============================================================
# UTILITIES
# ============================================================

def safe_parse_phi3(output: str):
    try:
        # extract ONLY JSON block
        json_match = re.search(r"\{.*\}", output, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found")

        return json.loads(json_match.group())

    except:
        return {
            "intent": "activity_summary",
            "activity": None,
            "time": {}
        }

def _canonical(activity: str) -> str:
    return activity.strip().upper().replace(" ", "_")


def _label(activity: str) -> str:
    return activity.replace("_", " ").lower()


def _format_clock(t: float) -> str:
    if t < 86400:
        return f"{t:g}"
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%H:%M:%S")


# ============================================================
# TIMELINE
# ============================================================

def build_timeline(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []

    for p in predictions:
        start = float(p["start_timestamp"])
        end = float(p["end_timestamp"])

        if end < start:
            continue

        out.append({
            "activity": _canonical(str(p["activity"])),
            "start": start,
            "end": end,
            "start_time": _format_clock(start),
            "end_time": _format_clock(end),
            "duration": end - start
        })

    return sorted(out, key=lambda x: x["start"])


def filter_time(timeline, t):
    if not t:
        return timeline

    start = t.get("from")
    end = t.get("to")

    if start is None and t.get("at") is not None:
        start = t["at"]
        end = t["at"]

    if start is None and end is None:
        return timeline

    out = []

    for s in timeline:
        if s["end"] < start or s["start"] > end:
            continue

        clip = dict(s)
        clip["start"] = max(s["start"], start)
        clip["end"] = min(s["end"], end)
        clip["duration"] = clip["end"] - clip["start"]
        out.append(clip)

    return out


# ============================================================
# ENGINE LOGIC (unchanged core, now driven by intent)
# ============================================================

def answer_intent(intent_data: Dict[str, Any], timeline: List[Dict[str, Any]]):

    intent = intent_data.get("intent")
    activity = intent_data.get("activity")
    time = intent_data.get("time", {})

    timeline = filter_time(timeline, time)

    # --------------------------------------------------------
    # 1. activity at time
    # --------------------------------------------------------
    if intent == "activity_at_time":
        t = time.get("at")
        match = next(
            (s for s in timeline if s["start"] <= t <= s["end"]),
            None
        )

        if match:
            return {
                "answer": f"You were {_label(match['activity'])} at {t}s",
                "evidence": [match]
            }

        return {"answer": "No activity found", "evidence": []}

    # --------------------------------------------------------
    # 2. duration
    # --------------------------------------------------------
    if intent == "activity_duration":
        matches = [s for s in timeline if s["activity"] == activity]
        total = sum(m["duration"] for m in matches)

        return {
            "answer": f"Total time: {round(total/60,2)} minutes {_label(activity)}",
            "evidence": matches
        }

    # --------------------------------------------------------
    # 3. count
    # --------------------------------------------------------
    if intent == "activity_count":
        matches = [s for s in timeline if s["activity"] == activity]

        return {
            "answer": f"{len(matches)} segments of {_label(activity)}",
            "evidence": matches
        }

    # --------------------------------------------------------
    # 4. time range / occurrences
    # --------------------------------------------------------
    if intent == "activity_time_range":
        matches = [s for s in timeline if s["activity"] == activity]

        spans = ", ".join(f"{m['start_time']}-{m['end_time']}" for m in matches)

        return {
            "answer": f"{_label(activity)} occurred during: {spans}",
            "evidence": matches
        }

    # --------------------------------------------------------
    # 5. summary
    # --------------------------------------------------------
    totals = {}
    for s in timeline:
        totals[s["activity"]] = totals.get(s["activity"], 0) + s["duration"]

    if totals:
        top = max(totals, key=totals.get)

        return {
            "answer": f"Most activity: {_label(top)} ({round(totals[top]/60,2)} min)",
            "evidence": [s for s in timeline if s["activity"] == top]
        }

    return {"answer": "No data", "evidence": []}


# ============================================================
# JSON LOADER
# ============================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_timeline(path):
    return build_timeline(load_json(path))


# ============================================================
# CLI
# ============================================================

def run_cli(json_path: str):
    timeline = load_timeline(json_path)

    print("\nIntent-Based Activity Engine (Phi-3)")
    print("Type 'exit' to quit\n")

    while True:
        q = input("Query: ")
        if q.strip().lower() == "exit":
            break

        intent = call_phi3(q)
        res = answer_intent(intent, timeline)

        print("\nAnswer:", res["answer"])
        print("Evidence:", len(res["evidence"]))
        print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python intent_activity_engine.py <json_file>")
    else:
        run_cli(sys.argv[1])