from lorascan.evaluate import GSM8KEvaluator, MultipleChoiceEvaluator, _fmt_gsm8k, _fmt_mc, get_evaluator

g, mc = GSM8KEvaluator(), MultipleChoiceEvaluator()


def test_gsm8k_extraction():
    assert g.extract("so 3*4=12\n#### 1,234") == "1234"
    assert g.extract("Thus the final answer is $72.50.") == "72.5"
    assert g.extract("It costs 5 then 18 dollars") == "18"
    assert g.extract("no numbers") is None
    assert g.score("The final answer is 72.", "72")


def test_mc_extraction():
    assert mc.extract("Reasoning... The answer is (C).") == "C"
    assert mc.extract("B") == "B"
    assert mc.extract("I think A is wrong. Final: D") == "D"


def test_gsm8k_formatter_roundtrips_through_evaluator():
    item = _fmt_gsm8k({"question": "q", "answer": "2+2=<<2+2=4>>4\n#### 4"}, {})
    assert item["gold"] == "4" and "<<" not in item["target"]
    assert g.score(item["target"], item["gold"])


def test_mc_formatter():
    item = _fmt_mc({"question": "q", "choices": ["w", "x", "y", "z"], "answer": 2}, {})
    assert item["gold"] == "C" and "C. y" in item["question"]
    assert mc.score(item["target"], "C")


def test_evaluate_aggregate_and_per_item():
    preds = [{"item_id": "a", "completion": "#### 4", "gold": "4", "completion_tokens": 10},
             {"item_id": "b", "completion": "#### 5", "gold": "4", "completion_tokens": 20},
             {"item_id": "c", "completion": "idk", "gold": "4", "completion_tokens": 30}]
    ev = get_evaluator("gsm8k")
    m = ev.evaluate(preds)
    assert (m["n"], m["n_correct"], m["n_unparsed"], m["mean_completion_tokens"]) == (3, 1, 1, 20)
    assert [r["correct"] for r in ev.per_item(preds)] == [True, False, False]
