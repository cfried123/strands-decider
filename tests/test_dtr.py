import json

import pytest

from strands_decider import dtr_eval
from strands_decider.data import dtr, recipes


@pytest.mark.parametrize("task", dtr.ALL_TASKS)
def test_generated_examples_are_valid(task):
    exs = dtr.generate(task, 30)
    assert exs
    for ex in exs:
        assert 0 <= ex.label < len(ex.options)
        assert ex.state["scenario"] and ex.state["dtr_citation"]
        ex.to_question()


def test_ambiguous_noul_rows_are_balanced_soft_targets():
    exs = dtr.generate("dtr_cpp_exception", 200)
    by_scenario = {}
    for ex in exs:
        by_scenario.setdefault(ex.state["scenario"], []).append(ex)
    pairs = [v for v in by_scenario.values() if len(v) > 1]
    assert pairs
    for pair in pairs:
        assert sorted(e.label for e in pair) == [0, 1]
        assert all(e.weight == 0.5 for e in pair)


def test_local_recipes_registered_and_build_without_hf():
    for task in dtr.ALL_TASKS:
        assert recipes.SPECS[task].local
    exs = recipes.build_recipe("dtr_travel_mode", max_examples=20)
    assert len(exs) == 20


def test_holdout_tasks_absent_from_routing_scenarios():
    routing = dtr.generate(dtr.ROUTING_TASK, 200)
    holdout = {
        s for t in dtr.HOLDOUT_TASKS for s in (e.state["scenario"] for e in dtr.generate(t, 100))
    }
    assert not holdout & {e.state["scenario"] for e in routing}


def test_bench_is_disjoint_from_training_and_sized():
    items = dtr.bench_items()
    assert 100 <= len(items) <= 200
    assert len({i["item_id"] for i in items}) == len(items)
    train = {e.state["scenario"] for t in dtr.ALL_TASKS for e in dtr.generate(t, 400)}
    assert not train & {i["state"]["scenario"] for i in items}
    assert {i["expected"] for i in items} >= {"abstain", "true", "false"}
    assert all(i["review"]["sme_review"] == "pending" for i in items)
    json.dumps(items)


def test_decide_applies_review_band():
    assert dtr_eval.decide({"type": "noul", "noul": 0.6})[0] == "abstain"
    assert dtr_eval.decide({"type": "noul", "noul": 0.9})[0] == "true"
    choice = {"type": "choice", "probabilities": {"a": 0.4, "b": 0.35, "c": 0.25}}
    assert dtr_eval.decide(choice) == ("abstain", "a", 0.4)
    score = {"type": "score", "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1}}
    assert dtr_eval.decide(score)[0] == "1"


def test_score_counts_correct_abstentions():
    items = [
        {"item_id": "a", "split": "in_domain", "kind": "noul", "task": "t", "expected": "true"},
        {"item_id": "b", "split": "in_domain", "kind": "noul", "task": "t", "expected": "abstain"},
    ]
    preds = {
        "a": {"decision": "true", "argmax": "true", "confidence": 0.9},
        "b": {"decision": "abstain", "argmax": "false", "confidence": 0.55},
    }
    m = dtr_eval.score(items, preds)["all"]
    assert m["strict_accuracy"] == 1.0
    assert m["abstain_recall"] == 1.0
    assert m["coverage"] == 1.0


def test_catalog_provisions_cover_every_bench_citation():
    cat = dtr.catalog()["tasks"]
    for it in dtr.bench_items():
        if it["task"] not in cat:
            continue
        cites = {p["dtr_citation"] for p in cat[it["task"]]["provisions"]}
        assert it["state"]["dtr_citation"] in cites, it["item_id"]
    booking = [p["section"] for p in cat["dtr_booking_policy"]["provisions"]]
    assert booking[0] == cat["dtr_booking_policy"]["section"] and "102.B" in booking
