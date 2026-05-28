from __future__ import annotations

from relbench_json2vec.materialize import materialize_split
from tests.fakes import make_binary_task


def test_materialize_filters_future_rows_and_caps_history() -> None:
    task = make_binary_task()
    split = materialize_split(
        dataset_name="fake",
        task_name="binary",
        task=task,
        split="val",
        history_limit=1,
        relation_depth=1,
        include_target=True,
    )

    first = split.records[0]
    assert first["__target__"] == 1
    assert len(first["relations"]["results"]) == 1
    assert first["relations"]["results"][0]["date"] <= first["__meta__"]["timestamp"]
    assert first["relations"]["results"][0]["position"] == 2


def test_materialize_omits_test_target() -> None:
    task = make_binary_task()
    split = materialize_split(
        dataset_name="fake",
        task_name="binary",
        task=task,
        split="test",
        history_limit=3,
        relation_depth=1,
        include_target=False,
    )
    assert "__target__" not in split.records[0]
