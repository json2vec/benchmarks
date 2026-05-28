from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from relbench_json2vec.materialize import materialize_split
from relbench_json2vec.schema import build_schema
from tests.fakes import FakeTaskType, make_binary_task, make_regression_task


@dataclass
class FakeField:
    name: str
    query: str | None = None
    type: str = ""
    p_prune: float | None = None
    topk: list[int] = field(default_factory=list)
    fields: list[Any] = field(default_factory=list)
    max_length: int = 1


class FakeJ2V:
    @staticmethod
    def _field(name: str, field_type: str, **kwargs: Any) -> FakeField:
        target = kwargs.pop("target", False)
        p_prune = 1.0 if target else kwargs.pop("p_prune", None)
        topk = kwargs.pop("topk", [])
        query = kwargs.pop("query", None)
        return FakeField(name=name, type=field_type, query=query, p_prune=p_prune, topk=topk)

    @staticmethod
    def Category(name: str, **kwargs: Any) -> FakeField:
        return FakeJ2V._field(name, "category", **kwargs)

    @staticmethod
    def Number(name: str, **kwargs: Any) -> FakeField:
        return FakeJ2V._field(name, "number", **kwargs)

    @staticmethod
    def DateParts(name: str, **kwargs: Any) -> FakeField:
        return FakeJ2V._field(name, "dateparts", **kwargs)

    @staticmethod
    def Set(name: str, **kwargs: Any) -> FakeField:
        return FakeJ2V._field(name, "set", **kwargs)

    @staticmethod
    def Array(*children: Any, name: str, max_length: int = 1, **kwargs: Any) -> FakeField:
        return FakeField(name=name, type="array", fields=list(children), max_length=max_length)

    @staticmethod
    def Address(*parts: str) -> tuple[str, ...]:
        return tuple(parts)


j2v = FakeJ2V()


def test_schema_infers_binary_target_and_datetime_field() -> None:
    task = make_binary_task()
    records = materialize_split(
        dataset_name="fake",
        task_name="binary",
        task=task,
        split="train",
        history_limit=3,
        relation_depth=1,
        include_target=True,
    ).records
    schema = build_schema(
        j2v=j2v,
        records=records,
        task_type=FakeTaskType.BINARY_CLASSIFICATION,
        history_limit=3,
        max_cat_vocab_size=32,
    )

    target = [field for field in schema.fields if field.name == "__target__"][0]
    assert target.type == "category"
    assert target.p_prune == 1.0
    assert target.topk == [2]
    relation = [field for field in schema.fields if field.name == "results"][0]
    assert any(child.name == "date" and child.type == "dateparts" for child in relation.fields)


def test_schema_infers_regression_target() -> None:
    task = make_regression_task()
    records = materialize_split(
        dataset_name="fake",
        task_name="regression",
        task=task,
        split="train",
        history_limit=3,
        relation_depth=1,
        include_target=True,
    ).records
    schema = build_schema(
        j2v=j2v,
        records=records,
        task_type=FakeTaskType.REGRESSION,
        history_limit=3,
        max_cat_vocab_size=32,
    )
    target = [field for field in schema.fields if field.name == "__target__"][0]
    assert target.type == "number"
    assert target.p_prune == 1.0
