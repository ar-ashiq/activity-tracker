from server.activity_engine.timeline import filter_time


def answer(intent_obj, timeline):

    intent = intent_obj["intent"]
    activity = intent_obj.get("activity")
    compare_activity = intent_obj.get("compare_activity")
    summary_type = intent_obj.get("summary_type", "most")
    time = intent_obj.get("time", {})

    if intent == "out_of_scope":
        return {
            "answer": "I can only answer questions about the recorded activities, such as what you were doing at a time, how long you walked, or which activity occurred most or least.",
            "evidence": [],
            "summary": {},
            "timeline_context": {}
        }

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
    selected_activity = dominant
    if summary_type == "least" and totals:
        selected_activity = min(totals, key=totals.get)

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
        if t is None:
            return {
                "answer": "A timestamp is required",
                "evidence": [],
                "summary": summary,
                "timeline_context": {}
            }

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
        if activity is None:
            return {
                "answer": "An activity is required",
                "evidence": [],
                "summary": summary,
                "timeline_context": {}
            }
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
        if activity is None:
            return {
                "answer": "An activity is required",
                "evidence": [],
                "summary": summary,
                "timeline_context": {}
            }
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
        if activity is None:
            return {
                "answer": "An activity is required",
                "evidence": [],
                "summary": summary,
                "timeline_context": {}
            }
        matches = [s for s in enriched if s["activity"] == activity]

        spans = [
            {
                "start": s["start"],
                "end": s["end"],
                "duration": s["duration_s"]
            }
            for s in matches
        ]

        if not spans:
            answer_text = f"No {activity.lower()} activity was found."
        else:
            formatted_spans = ", ".join(
                f"{span['start']:g}s-{span['end']:g}s" for span in spans
            )
            answer_text = f"Yes, the user was {activity.lower()} at: {formatted_spans}."

        return {
            "answer": answer_text,
            "evidence": spans,
            "summary": summary,
            "timeline_context": {
                "total_time": totals.get(activity, 0)
            }
        }

    # ----------------------------------------------------
    # 5. PRESENCE
    # ----------------------------------------------------
    if intent == "activity_presence":
        if activity is None:
            return {
                "answer": "An activity is required",
                "evidence": [],
                "summary": summary,
                "timeline_context": {}
            }

        matches = [s for s in enriched if s["activity"] == activity]
        present = bool(matches)

        return {
            "answer": f"Yes, the user was {activity.lower()}." if present else f"No, the user was not {activity.lower()}.",
            "evidence": matches,
            "summary": summary,
            "timeline_context": {
                "activity_present": present,
                "segment_count": len(matches),
                "total_time": totals.get(activity, 0)
            }
        }

    # ----------------------------------------------------
    # 6. COMPARE ACTIVITIES
    # ----------------------------------------------------
    if intent == "activity_compare":
        if activity is None or compare_activity is None:
            return {
                "answer": "Two activities are required",
                "evidence": [],
                "summary": summary,
                "timeline_context": {}
            }

        first_total = totals.get(activity, 0)
        second_total = totals.get(compare_activity, 0)
        if first_total == second_total:
            answer_text = (
                f"The user spent the same amount of time in "
                f"{activity.lower()} and {compare_activity.lower()}."
            )
            selected = [activity, compare_activity]
        else:
            winner = activity if first_total > second_total else compare_activity
            loser = compare_activity if winner == activity else activity
            winner_total = max(first_total, second_total)
            answer_text = (
                f"The user spent more time in {winner.lower()} than "
                f"{loser.lower()} ({round(winner_total / 60, 2)} min)."
            )
            selected = [winner]

        return {
            "answer": answer_text,
            "evidence": [s for s in enriched if s["activity"] in selected],
            "summary": summary,
            "timeline_context": {
                "durations_sec": {
                    activity: first_total,
                    compare_activity: second_total,
                }
            }
        }

    # ----------------------------------------------------
    # 7. SUMMARY
    # ----------------------------------------------------
    return {
        "answer": f"{summary_type.capitalize()} activity: {selected_activity.lower() if selected_activity else 'none'}",
        "evidence": [s for s in enriched if s["activity"] == selected_activity],
        "summary": summary,
        "timeline_context": {
            "transitions": transitions,
            "activity_distribution": totals,
            "summary_type": summary_type,
        }
    }