"""Small, explicit end-to-end smoke set. Run inside the model container."""

import json
import time

from jasmine_local import LlamaBackend, evaluate


CASES = [
    ("I was charged twice and need a refund.", "Which team handles this?", {"billing": "Payments and refunds", "technical": "Software bugs", "sales": "Pricing and new purchases"}, "billing"),
    ("The app crashes every time I tap export.", "Which team handles this?", {"billing": "Payments and refunds", "technical": "Software bugs", "sales": "Pricing and new purchases"}, "technical"),
    ("I'd like pricing for twenty seats.", "Which team handles this?", {"billing": "Payments and refunds", "technical": "Software bugs", "sales": "Pricing and new purchases"}, "sales"),
    ("I'd like pricing for twenty seats.", "Which team handles this?", {"sales": "Pricing and new purchases", "billing": "Payments and refunds", "technical": "Software bugs"}, "sales"),
    ("Please refund the duplicate charge.", "Which team handles this?", {"technical": "Software bugs", "sales": "Pricing and new purchases", "billing": "Payments and refunds"}, "billing"),
]


def main():
    backend = LlamaBackend()
    correct = 0
    elapsed = []
    distributions = []
    for state, instructions, criteria, expected in CASES:
        started = time.monotonic()
        result = evaluate({"model": "jasmine-qwen3.5-0.8b", "state": state, "questions": {"route": {"type": "choice", "instructions": instructions, "criteria": criteria}}}, backend)
        elapsed.append(round((time.monotonic() - started) * 1000))
        answer = result["answers"]["route"]
        distributions.append(answer["probabilities"])
        correct += answer["choice"] == expected
        print(json.dumps({"expected": expected, "got": answer["choice"], "probabilities": answer["probabilities"], "ms": elapsed[-1]}))
    print(json.dumps({"correct": correct, "total": len(CASES), "median_ms": sorted(elapsed)[len(elapsed) // 2]}))
    order_delta = max(abs(distributions[2][key] - distributions[3][key]) for key in distributions[2])
    print(json.dumps({"reordered_choice_max_probability_delta": order_delta}))
    pair = []
    for state in ("No deadline; I can wait until next week.", "I need this fixed today before my flight."):
        result = evaluate({"model": "jasmine-qwen3.5-0.8b", "state": state, "questions": {
            "urgent": {"type": "noul", "instructions": "Does the customer explicitly need help today?"},
            "severity": {"type": "score", "instructions": "How urgent is the request?", "criteria": ["No deadline", "Needed within a week", "Needed today"]},
        }}, backend)
        answers = result["answers"]
        pair.append((answers["urgent"]["noul"], answers["severity"]["score"]))
        print(json.dumps({"state": state, "noul": pair[-1][0], "score": pair[-1][1]}))
    monotonic = pair[1][0] > pair[0][0] and pair[1][1] > pair[0][1]
    print(json.dumps({"urgency_monotonic": monotonic}))
    if correct != len(CASES) or not monotonic or order_delta > 1e-6:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
