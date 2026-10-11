import requests


def call_qwen(question: str):

    system_prompt = """
You classify a user's activity question into exactly one predefined intent.
Return ONLY one valid JSON object. Do not include markdown or explanations.

Allowed intents and selection rules:
- activity_at_time: asks what activity happened at one timestamp. Set time.at.
- activity_duration: asks how long an activity lasted. Set activity.
- activity_time_range: asks when an activity occurred, including questions such as "did I run at any point, and if so when?" Set activity.
- activity_count: asks how many occurrences or segments an activity has. Set activity.
- activity_presence: asks whether an activity occurred at all. Set activity.
- activity_compare: compares which of two named activities took more or less time. Set activity and compare_activity.
- activity_summary: asks which activity occurred the most or least. Set summary_type to "most" or "least".
- out_of_scope: greetings, general conversation, or questions unrelated to recorded activities.

Allowed activities:
WALKING, RUNNING, SITTING, STANDING, LYING_DOWN, BICYCLING, null

The response schema is:
{
    "intent": "activity_at_time | activity_duration | activity_time_range | activity_count | activity_presence | activity_compare | activity_summary | out_of_scope",
    "activity": "WALKING | RUNNING | SITTING | STANDING | LYING_DOWN | BICYCLING | null",
    "compare_activity": "WALKING | RUNNING | SITTING | STANDING | LYING_DOWN | BICYCLING | null",
    "summary_type": "most | least",
    "time": {"at": number or null, "from": number or null, "to": number or null}
}

Examples:
Question: What was I doing at 600 seconds?
JSON: {"intent":"activity_at_time","activity":null,"time":{"at":600,"from":null,"to":null}}
Question: How long did I walk?
JSON: {"intent":"activity_duration","activity":"WALKING","time":{"at":null,"from":null,"to":null}}
Question: When was I sitting?
JSON: {"intent":"activity_time_range","activity":"SITTING","time":{"at":null,"from":null,"to":null}}
Question: Did the user run at any point? If so, when?
JSON: {"intent":"activity_time_range","activity":"RUNNING","time":{"at":null,"from":null,"to":null}}
Question: How many times did I run?
JSON: {"intent":"activity_count","activity":"RUNNING","time":{"at":null,"from":null,"to":null}}
Question: Is the user walking?
JSON: {"intent":"activity_presence","activity":"WALKING","time":{"at":null,"from":null,"to":null}}
Question: Did the user spend most time running or walking?
JSON: {"intent":"activity_compare","activity":"RUNNING","compare_activity":"WALKING","time":{"at":null,"from":null,"to":null}}
Question: What was the user's main activity?
JSON: {"intent":"activity_summary","activity":null,"summary_type":"most","time":{"at":null,"from":null,"to":null}}
Question: What was the user doing the least?
JSON: {"intent":"activity_summary","activity":null,"summary_type":"least","time":{"at":null,"from":null,"to":null}}
Question: hi
JSON: {"intent":"out_of_scope","activity":null,"summary_type":null,"time":{"at":null,"from":null,"to":null}}

Use activity_summary only for questions asking which activity occurred the most or least. Use out_of_scope for everything else.
"""

    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:7b",
            "prompt": system_prompt + f"\n\nQuestion: {question}",
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.1
            }
        },
        timeout=60,
    )
    res.raise_for_status()

    return res.json()["response"]