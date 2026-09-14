import requests
import json
from server.activity_engine.config import VALID_INTENTS


def call_qwen(question: str):

    SYSTEM_PROMPT = """
You are an INTENT CLASSIFIER ONLY.

You do NOT reason.
You do NOT explain.
You do NOT use external knowledge.

Return ONLY valid JSON.

INTENTS:
- activity_at_time
- activity_duration
- activity_time_range
- activity_count
- activity_summary

ACTIVITIES:
WALKING, RUNNING, SITTING, STANDING, LYING_DOWN, BICYCLING, null

OUTPUT FORMAT:
{
  "intent": "string",
  "activity": "string or null",
  "time": {
    "at": number or null,
    "from": number or null,
    "to": number or null
  }
}

RULE:
Return ONLY JSON. No extra text.
"""

    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:7b",   # ✅ CHANGE HERE (Qwen instead of Phi-3)
            "prompt": SYSTEM_PROMPT + f"\n\nQuestion: {question}",
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.1
            }
        }
    )

    return res.json()["response"]