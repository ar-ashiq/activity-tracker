from server.activity_engine.timeline import filter_time


def answer(intent_obj, timeline):

    intent = intent_obj["intent"]
    activity = intent_obj.get("activity")
    time = intent_obj.get("time", {})

    timeline = filter_time(timeline, time)

    # ----------------------------------------------------
    # Helper: enrich segments
    # ----------------------------------------------------
    def enrich(seg):
        return {
            **seg,
            "duration_s": seg["end"] - seg["start"],
            "midpoint": (seg["start"] + seg["end"]) / 2
        }

    enriched = [enrich(s) for s in timeline]

    # ----------------------------------------------------
    # SUMMARY METADATA (GLOBAL)
    # ----------------------------------------------------
    totals = {}
    transitions = 0

    for i in range(len(enriched)):
        a = enriched[i]["activity"]
        totals[a] = totals.get(a, 0) + enriched[i]["duration_s"]

        if i > 0 and enriched[i]["activity"] != enriched[i-1]["activity"]:
            transitions += 1

    dominant = max(totals, key=totals.get) if totals else None

    summary = {
        "total_segments": len(enriched),
        "total_time_sec": sum(totals.values()) if totals else 0,
        "dominant_activity": dominant
    }

    # ----------------------------------------------------
    # 1. ACTIVITY AT TIME
    # ----------------------------------------------------
    if intent == "activity_at_time":
        t = time.get("at")

        match = next(
            (s for s in enriched if s["start"] <= t <= s["end"]),
            None
        )

        if not match:
            return {
                "answer": "No activity found",
                "evidence": [],
                "summary": summary,
                "timeline_context": {}
            }

        return {
            "answer": f"You were {match['activity'].lower()} at {t}s",
            "evidence": [match],
            "summary": summary,
            "timeline_context": {
                "previous_activity": enriched[enriched.index(match)-1]["activity"] if enriched.index(match) > 0 else None,
                "next_activity": enriched[enriched.index(match)+1]["activity"] if enriched.index(match)+1 < len(enriched) else None
            }
        }

    # ----------------------------------------------------
    # 2. DURATION
    # ----------------------------------------------------
    if intent == "activity_duration":
        matches = [s for s in enriched if s["activity"] == activity]

        total = sum(s["duration_s"] for s in matches)

        return {
            "answer": f"Total time: {round(total/60,2)} min {activity.lower()}",
            "evidence": matches,
            "summary": summary,
            "timeline_context": {
                "dominant_activity": dominant,
                "transitions": transitions
            }
        }

    # ----------------------------------------------------
    # 3. COUNT
    # ----------------------------------------------------
    if intent == "activity_count":
        matches = [s for s in enriched if s["activity"] == activity]

        return {
            "answer": f"{len(matches)} segments of {activity.lower()}",
            "evidence": matches,
            "summary": summary,
            "timeline_context": {
                "activity_share": (
                    totals.get(activity, 0) / summary["total_time_sec"]
                    if summary["total_time_sec"] else 0
                )
            }
        }

    # ----------------------------------------------------
    # 4. TIME RANGE
    # ----------------------------------------------------
    if intent == "activity_time_range":
        matches = [s for s in enriched if s["activity"] == activity]

        spans = [
            {
                "start": s["start"],
                "end": s["end"],
                "duration": s["duration_s"]
            }
            for s in matches
        ]

        return {
            "answer": f"{activity.lower()} occurred {len(matches)} times",
            "evidence": spans,
            "summary": summary,
            "timeline_context": {
                "total_time": totals.get(activity, 0)
            }
        }

    # ----------------------------------------------------
    # 5. SUMMARY
    # ----------------------------------------------------
    return {
        "answer": f"Most activity: {dominant.lower() if dominant else 'none'}",
        "evidence": enriched,
        "summary": summary,
        "timeline_context": {
            "transitions": transitions,
            "activity_distribution": totals
        }
    }