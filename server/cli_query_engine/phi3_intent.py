def call_phi3_intent(question: str):

    prompt = f"""
You are ONLY an intent classifier.

You DO NOT see any data.

Return ONLY valid JSON.

Allowed intents:
- activity_at_time
- activity_duration
- activity_time_range
- activity_count
- activity_summary

Allowed activities:
WALKING, RUNNING, SITTING, STANDING, LYING_DOWN, BICYCLING, null

OUTPUT FORMAT:
{{
  "intent": "...",
  "activity": "... or null",
  "time": {{
    "at": number or null,
    "from": number or null,
    "to": number or null
  }}
}}

RULES:
- NO explanations
- NO markdown
- ONLY JSON

Question:
{question}
"""

    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi3",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }
    )

    raw = res.json()["response"]
    return raw