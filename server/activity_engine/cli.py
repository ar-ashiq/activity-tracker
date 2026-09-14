import sys
from server.activity_engine.qwen import call_qwen
from server.activity_engine.sanitizer import sanitize
from server.activity_engine.engine import answer
from server.activity_engine.loader import load_timeline

def run(json_path):

    timeline = load_timeline(json_path)

    print("\nQwen2.5 Intent Activity Engine Ready")
    print("Type exit to quit\n")

    while True:
        q = input("Query: ")

        if q.lower() == "exit":
            break

        raw = call_qwen(q)
        intent = sanitize(raw)
        res = answer(intent, timeline)

        print("\nAnswer:", res["answer"])
        print("Evidence:", len(res["evidence"]))
        print()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python cli.py data.json")
    else:
        run(sys.argv[1])