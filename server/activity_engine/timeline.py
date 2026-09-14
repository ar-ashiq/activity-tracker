from datetime import datetime, timezone

def build_timeline(predictions):
    timeline = []

    for p in predictions:
        start = float(p["start_timestamp"])
        end = float(p["end_timestamp"])

        if end < start:
            continue

        timeline.append({
            "activity": p["activity"].strip().upper(),
            "start": start,
            "end": end,
            "start_time": _fmt(start),
            "end_time": _fmt(end)
        })

    return sorted(timeline, key=lambda x: x["start"])


def _fmt(t):
    if t < 86400:
        return f"{t:g}"
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%H:%M:%S")


def filter_time(timeline, time):
    if not time or (time.get("from") is None and time.get("to") is None and time.get("at") is None):
        return timeline

    start = time.get("from")
    end = time.get("to")

    if time.get("at") is not None:
        start = end = time["at"]

    return [
        s for s in timeline
        if not (s["end"] < start or s["start"] > end)
    ]