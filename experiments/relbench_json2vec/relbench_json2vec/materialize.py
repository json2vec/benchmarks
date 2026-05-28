from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z_]+")


@dataclass
class MaterializedSplit:
    split: str
    records: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)


def safe_name(value: Any) -> str:
    name = SAFE_NAME_RE.sub("_", str(value)).strip("_")
    if not name:
        name = "field"
    if name[0].isdigit():
        name = f"f_{name}"
    return name


def _dedupe_safe_names(columns: list[Any]) -> dict[Any, str]:
    used: dict[str, int] = {}
    out: dict[Any, str] = {}
    for column in columns:
        base = safe_name(column)
        count = used.get(base, 0)
        used[base] = count + 1
        out[column] = base if count == 0 else f"{base}_{count}"
    return out


def normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, set)) else False:
        return None
    if isinstance(value, dict):
        return {safe_name(key): normalize_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [normalize_value(inner) for inner in value]
    return value


def _row_to_dict(row: pd.Series, column_map: dict[Any, str], *, exclude: set[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for column, value in row.items():
        if column in exclude:
            continue
        out[column_map[column]] = normalize_value(value)
    return out


def _entity_record(db: Any, task: Any, entity_id: Any) -> dict[str, Any]:
    table = db.table_dict[task.entity_table]
    df = table.df
    column_map = _dedupe_safe_names(list(df.columns))
    pkey_col = table.pkey_col
    if pkey_col is None:
        return {}

    matched = df[df[pkey_col] == entity_id]
    if matched.empty:
        return {}
    return _row_to_dict(matched.iloc[0], column_map, exclude={pkey_col})


def _related_records(
    db: Any,
    task: Any,
    *,
    entity_id: Any,
    timestamp: Any,
    history_limit: int,
    relation_depth: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    if relation_depth < 1:
        return {}, []

    warnings: list[str] = []
    relations: dict[str, list[dict[str, Any]]] = {}
    for table_name, table in db.table_dict.items():
        if table_name == task.entity_table:
            continue
        fkeys = [
            column
            for column, pkey_table in table.fkey_col_to_pkey_table.items()
            if pkey_table == task.entity_table
        ]
        if not fkeys:
            continue

        df = table.df
        matched_parts = [df[df[fkey] == entity_id] for fkey in fkeys if fkey in df.columns]
        if not matched_parts:
            continue
        matched = pd.concat(matched_parts, axis=0).drop_duplicates()
        time_col = table.time_col
        if time_col is not None and time_col in matched.columns and timestamp is not None:
            matched = matched[matched[time_col].le(timestamp)]
            matched = matched.sort_values(time_col, ascending=False)
        if history_limit > 0:
            matched = matched.head(history_limit)

        column_map = _dedupe_safe_names(list(matched.columns))
        exclude = set(fkeys)
        if table.pkey_col is not None:
            exclude.add(table.pkey_col)
        relation_name = safe_name(table_name)
        relations[relation_name] = [
            _row_to_dict(row, column_map, exclude=exclude)
            for _, row in matched.iterrows()
        ]
        if len(matched) == history_limit:
            warnings.append(f"relation {table_name!r} was capped at {history_limit} rows")

    return relations, warnings


def materialize_split(
    *,
    dataset_name: str,
    task_name: str,
    task: Any,
    split: str,
    history_limit: int,
    relation_depth: int,
    include_target: bool,
) -> MaterializedSplit:
    table = task.get_table(split, mask_input_cols=not include_target)
    db = task.dataset.get_db(upto_test_timestamp=split != "test")
    entity_col = task.entity_col
    time_col = task.time_col
    target_col = task.target_col
    records: list[dict[str, Any]] = []
    warnings: list[str] = []

    for row_index, row in table.df.reset_index(drop=True).iterrows():
        entity_id = row[entity_col]
        timestamp = row[time_col] if time_col in row else None
        relations, relation_warnings = _related_records(
            db,
            task,
            entity_id=entity_id,
            timestamp=timestamp,
            history_limit=history_limit,
            relation_depth=relation_depth,
        )
        warnings.extend(relation_warnings)

        record = {
            "__meta__": {
                "dataset": dataset_name,
                "task": task_name,
                "split": split,
                "row_index": int(row_index),
                "timestamp": normalize_value(timestamp),
                "entity_table": task.entity_table,
                "entity_id": normalize_value(entity_id),
            },
            "entity": _entity_record(db, task, entity_id),
            "relations": relations,
        }
        if include_target and target_col in row:
            record["__target__"] = normalize_value(row[target_col])
        records.append(record)

    return MaterializedSplit(split=split, records=records, warnings=sorted(set(warnings)))


def materialize_task(
    *,
    dataset_name: str,
    task_name: str,
    task: Any,
    history_limit: int,
    relation_depth: int,
) -> dict[str, MaterializedSplit]:
    return {
        "train": materialize_split(
            dataset_name=dataset_name,
            task_name=task_name,
            task=task,
            split="train",
            history_limit=history_limit,
            relation_depth=relation_depth,
            include_target=True,
        ),
        "val": materialize_split(
            dataset_name=dataset_name,
            task_name=task_name,
            task=task,
            split="val",
            history_limit=history_limit,
            relation_depth=relation_depth,
            include_target=True,
        ),
        "test": materialize_split(
            dataset_name=dataset_name,
            task_name=task_name,
            task=task,
            split="test",
            history_limit=history_limit,
            relation_depth=relation_depth,
            include_target=False,
        ),
    }
