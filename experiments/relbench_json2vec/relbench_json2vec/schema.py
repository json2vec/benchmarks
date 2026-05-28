from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

@dataclass
class SchemaBuildResult:
    fields: list[Any]
    target_address: Any
    warnings: list[str] = field(default_factory=list)


def _iter_values(records: Iterable[dict[str, Any]], path: tuple[str, ...]) -> Iterable[Any]:
    for record in records:
        node: Any = record
        for part in path:
            if isinstance(node, list):
                for item in node:
                    if isinstance(item, dict) and part in item:
                        yield item[part]
                node = None
                break
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if node is not None:
            yield node


def _is_scalar_list(value: Any) -> bool:
    return isinstance(value, list) and all(not isinstance(item, (dict, list, tuple, set)) for item in value)


def _field_for_values(
    j2v: Any,
    *,
    name: str,
    query: str,
    values: list[Any],
    max_cat_vocab_size: int,
    target: bool = False,
    binary_target: bool = False,
) -> tuple[Any | None, str | None]:
    non_null = [value for value in values if value is not None]
    if not non_null:
        return None, f"skipped {query}: no non-null values"

    if all(_is_scalar_list(value) for value in non_null):
        return j2v.Set(name=name, query=query, max_vocab_size=max_cat_vocab_size), None
    if any(isinstance(value, (dict, list, tuple, set)) for value in non_null):
        return None, f"skipped {query}: nested non-scalar values are not supported in v1"

    sample = non_null[0]
    if isinstance(sample, pd.Timestamp) or "datetime" in type(sample).__name__.lower():
        return (
            j2v.DateParts(
                name=name,
                query=query,
                dateparts=["month_of_year", "day_of_month", "day_of_week"],
            ),
            None,
        )
    if isinstance(sample, bool):
        return (
            j2v.Category(
                name=name,
                query=query,
                max_vocab_size=max_cat_vocab_size,
                target=target,
                topk=[2] if binary_target else [],
            ),
            None,
        )
    if isinstance(sample, (int, float)) and not isinstance(sample, bool):
        return j2v.Number(name=name, query=query, target=target), None
    return (
        j2v.Category(
            name=name,
            query=query,
            max_vocab_size=max_cat_vocab_size,
            target=target,
            topk=[2] if binary_target else [],
        ),
        None,
    )


def _entity_fields(j2v: Any, records: list[dict[str, Any]], max_cat_vocab_size: int) -> tuple[list[Any], list[str]]:
    names = sorted({key for record in records for key in record.get("entity", {}).keys()})
    fields: list[Any] = []
    warnings: list[str] = []
    for name in names:
        query = f"[*].entity.{name}"
        field, warning = _field_for_values(
            j2v,
            name=name,
            query=query,
            values=list(_iter_values(records, ("entity", name))),
            max_cat_vocab_size=max_cat_vocab_size,
        )
        if field is not None:
            fields.append(field)
        if warning is not None:
            warnings.append(warning)
    return fields, warnings


def _relation_fields(
    j2v: Any,
    records: list[dict[str, Any]],
    *,
    history_limit: int,
    max_cat_vocab_size: int,
) -> tuple[list[Any], list[str]]:
    relation_names = sorted(
        {
            relation_name
            for record in records
            for relation_name in record.get("relations", {}).keys()
        }
    )
    arrays: list[Any] = []
    warnings: list[str] = []
    for relation_name in relation_names:
        child_names = sorted(
            {
                key
                for record in records
                for row in record.get("relations", {}).get(relation_name, [])
                for key in row.keys()
            }
        )
        children: list[Any] = []
        for child_name in child_names:
            query = f"[*].relations.{relation_name}[*].{child_name}"
            values = [
                row.get(child_name)
                for record in records
                for row in record.get("relations", {}).get(relation_name, [])
            ]
            field, warning = _field_for_values(
                j2v,
                name=child_name,
                query=query,
                values=values,
                max_cat_vocab_size=max_cat_vocab_size,
            )
            if field is not None:
                children.append(field)
            if warning is not None:
                warnings.append(warning)
        if children:
            arrays.append(
                j2v.Array(
                    *children,
                    name=relation_name,
                    max_length=max(1, history_limit),
                    n_outputs=1,
                )
            )
    return arrays, warnings


def build_schema(
    *,
    j2v: Any,
    records: list[dict[str, Any]],
    task_type: Any,
    history_limit: int,
    max_cat_vocab_size: int,
) -> SchemaBuildResult:
    task_type_value = getattr(task_type, "value", str(task_type))
    warnings: list[str] = []
    entity_fields, entity_warnings = _entity_fields(j2v, records, max_cat_vocab_size)
    relation_fields, relation_warnings = _relation_fields(
        j2v,
        records,
        history_limit=history_limit,
        max_cat_vocab_size=max_cat_vocab_size,
    )
    warnings.extend(entity_warnings)
    warnings.extend(relation_warnings)

    if task_type_value == "binary_classification":
        target_field = j2v.Category(
            "__target__",
            query="[*].__target__",
            target=True,
            max_vocab_size=max(max_cat_vocab_size, 8),
            topk=[2],
        )
    elif task_type_value == "regression":
        target_field = j2v.Number("__target__", query="[*].__target__", target=True)
    else:
        raise ValueError(f"unsupported RelBench task type for v1: {task_type_value}")

    fields: list[Any] = [
        j2v.Array(*entity_fields, name="entity", max_length=1, n_outputs=1),
        *relation_fields,
        target_field,
    ]
    if not entity_fields and not relation_fields:
        warnings.append("schema has no non-target input fields")

    return SchemaBuildResult(
        fields=fields,
        target_address=j2v.Address("record", "__target__"),
        warnings=sorted(set(warnings)),
    )
