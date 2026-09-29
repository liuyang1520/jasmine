import unittest

from jasmine_local import InputError, LlamaBackend, evaluate, validate_request


class FakeBackend:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def decide(self, state, instructions, options):
        return self.probabilities.pop(0), 10, 1


class LocalDecisionTests(unittest.TestCase):
    def test_three_typed_answers_are_computed_from_probabilities(self):
        payload = {
            "model": "jev-latest",
            "state": {"message": "charged twice"},
            "questions": {
                "route": {"type": "choice", "instructions": "Where?", "criteria": {"billing": None, "sales": None}},
                "urgent": {"type": "noul", "instructions": "Urgent?"},
                "severity": {"type": "score", "instructions": "How severe?", "criteria": ["low", "mid", "high"]},
            },
        }
        answer = evaluate(payload, FakeBackend([
            [0.8, 0.2], [0.2, 0.8],
            [0.3, 0.7], [0.7, 0.3],
            [0.1, 0.2, 0.7], [0.2, 0.7, 0.1], [0.7, 0.1, 0.2],
        ]))
        self.assertEqual(answer["answers"]["route"]["choice"], "billing")
        self.assertAlmostEqual(answer["answers"]["urgent"]["noul"], 0.7)
        self.assertAlmostEqual(answer["answers"]["severity"]["score"], 1.6)
        self.assertEqual(answer["usage"], {"input_tokens": 70, "output_tokens": 7})

    def test_choice_order_does_not_change_semantic_probabilities(self):
        class BiasedBackend:
            def decide(self, state, instructions, options):
                weights = [(4 if "sales" in option else 1) + (2 if i == 0 else 0)
                           for i, option in enumerate(options)]
                return [weight / sum(weights) for weight in weights], 10, 1

        def ask(criteria):
            payload = {"model": "jev-latest", "state": "pricing", "questions": {
                "route": {"type": "choice", "instructions": "Which team?", "criteria": criteria}}}
            return evaluate(payload, BiasedBackend())["answers"]["route"]

        first = ask({"billing": None, "sales": None, "technical": None})
        second = ask({"technical": None, "billing": None, "sales": None})
        self.assertEqual(first["choice"], second["choice"])
        for key in first["probabilities"]:
            self.assertAlmostEqual(first["probabilities"][key], second["probabilities"][key])

    def test_rejects_extra_fields_and_more_than_ten_choices(self):
        base = {"model": "jev-latest", "state": "x", "questions": {"q": {"type": "noul", "instructions": "Is it?"}}}
        validate_request(base)
        with self.assertRaises(InputError):
            validate_request({**base, "backend_url": "http://example.com"})
        with self.assertRaises(InputError):
            validate_request({**base, "questions": {"q": {"type": "choice", "instructions": "Which?", "criteria": {str(i): None for i in range(11)}}}})

    def test_backend_url_cannot_use_userinfo_to_escape_loopback(self):
        with self.assertRaises(ValueError):
            LlamaBackend("http://127.0.0.1:8080@evil.example")


if __name__ == "__main__":
    unittest.main()
