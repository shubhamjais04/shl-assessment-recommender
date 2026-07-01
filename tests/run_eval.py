"""
Run this AFTER deploying to Render (or against localhost:8000 for a local
test with a real GROQ_API_KEY set).

Usage:
    python tests/run_eval.py --url http://localhost:8000
    python tests/run_eval.py --url https://your-app.onrender.com

Replays each trace's user turns in order against /chat (building up the
conversation history exactly as the real evaluator would), then compares
the final turn's recommendations against the trace's expected shortlist to
compute Recall@10. Also checks basic schema compliance on every response.
"""

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.trace_parser import load_all_traces


def normalize(name: str) -> str:
    return name.lower().strip().replace("–", "-").replace("  ", " ")


def run_trace(base_url, trace, timeout=30):
    messages = []
    final_recs = []
    for user_turn in trace["user_turns"]:
        messages.append({"role": "user", "content": user_turn})
        resp = requests.post(f"{base_url}/chat", json={"messages": messages}, timeout=timeout)
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
        data = resp.json()

        # basic schema check
        for key in ("reply", "recommendations", "end_of_conversation"):
            if key not in data:
                return None, f"Missing key '{key}' in response"
        if not (0 <= len(data["recommendations"]) <= 10):
            return None, f"recommendations length out of bounds: {len(data['recommendations'])}"

        messages.append({"role": "assistant", "content": data["reply"]})
        if data["recommendations"]:
            final_recs = data["recommendations"]

    return final_recs, None


def recall_at_10(predicted_names, expected_names):
    if not expected_names:
        return None
    pred_norm = {normalize(n) for n in predicted_names}
    exp_norm = {normalize(n) for n in expected_names}
    hit = len(pred_norm & exp_norm)
    return hit / len(exp_norm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Base URL of the deployed service")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    print(f"Checking /health at {base_url} ...")
    try:
        h = requests.get(f"{base_url}/health", timeout=120)
        print("  ->", h.status_code, h.json())
    except Exception as e:
        print("  Health check failed:", e)
        return

    traces = load_all_traces()
    recalls = []
    print(f"\nReplaying {len(traces)} traces...\n")

    for trace in traces:
        recs, error = run_trace(base_url, trace)
        if error:
            print(f"[{trace['file']}] ERROR: {error}")
            continue
        predicted_names = [r["name"] for r in recs]
        recall = recall_at_10(predicted_names, trace["expected_shortlist"])
        recalls.append(recall)
        print(f"[{trace['file']}] Recall@10 = {recall:.2f}")
        print(f"    expected : {trace['expected_shortlist']}")
        print(f"    predicted: {predicted_names}")
        print()

    if recalls:
        mean_recall = sum(recalls) / len(recalls)
        print(f"\n=== Mean Recall@10 across {len(recalls)} traces: {mean_recall:.3f} ===")


if __name__ == "__main__":
    main()
