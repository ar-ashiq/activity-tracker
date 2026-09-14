import json
from server.activity_engine.timeline import build_timeline


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_timeline(path):
    return build_timeline(load_json(path))