from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class FakeTaskType(Enum):
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"


@dataclass
class FakeTable:
    df: pd.DataFrame
    fkey_col_to_pkey_table: dict[str, str]
    pkey_col: str | None = None
    time_col: str | None = None

    def __len__(self) -> int:
        return len(self.df)


@dataclass
class FakeDatabase:
    table_dict: dict[str, FakeTable]


class FakeDataset:
    def __init__(self) -> None:
        drivers = FakeTable(
            df=pd.DataFrame(
                {
                    "driverId": [0, 1, 2],
                    "age": [30, 40, 35],
                    "country": ["uk", "de", "br"],
                }
            ),
            fkey_col_to_pkey_table={},
            pkey_col="driverId",
        )
        results = FakeTable(
            df=pd.DataFrame(
                {
                    "resultId": [0, 1, 2, 3, 4, 5],
                    "driverId": [0, 0, 0, 1, 1, 2],
                    "date": pd.to_datetime(
                        [
                            "2020-01-01",
                            "2020-01-05",
                            "2020-03-01",
                            "2020-01-03",
                            "2020-02-10",
                            "2020-01-04",
                        ]
                    ),
                    "position": [1, 2, 8, 3, 5, 4],
                    "status": ["ok", "ok", "dnf", "ok", "dnf", "ok"],
                }
            ),
            fkey_col_to_pkey_table={"driverId": "drivers"},
            pkey_col="resultId",
            time_col="date",
        )
        self.db = FakeDatabase({"drivers": drivers, "results": results})

    def get_db(self, upto_test_timestamp: bool = True) -> FakeDatabase:
        return self.db


class FakeEntityTask:
    entity_col = "driverId"
    entity_table = "drivers"
    time_col = "date"
    target_col = "target"

    def __init__(self, *, task_type: FakeTaskType, train_targets: list[Any], val_targets: list[Any], test_targets: list[Any]):
        self.dataset = FakeDataset()
        self.task_type = task_type
        self._tables = {
            "train": self._make_table("train", ["2020-01-10", "2020-01-20", "2020-01-30"], [0, 1, 2], train_targets),
            "val": self._make_table("val", ["2020-02-01", "2020-02-05"], [0, 1], val_targets),
            "test": self._make_table("test", ["2020-02-15", "2020-02-20"], [0, 1], test_targets),
        }

    def _make_table(self, split: str, dates: list[str], ids: list[int], targets: list[Any]) -> FakeTable:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(dates),
                "driverId": ids,
                "target": targets,
            }
        )
        return FakeTable(
            df=df,
            fkey_col_to_pkey_table={"driverId": "drivers"},
            pkey_col=None,
            time_col="date",
        )

    def get_table(self, split: str, mask_input_cols: bool | None = None) -> FakeTable:
        table = self._tables[split]
        df = table.df.copy()
        if mask_input_cols is None:
            mask_input_cols = split == "test"
        if mask_input_cols and "target" in df:
            df = df.drop(columns=["target"])
        return FakeTable(
            df=df,
            fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
            pkey_col=table.pkey_col,
            time_col=table.time_col,
        )

    def evaluate(self, pred: np.ndarray, target_table: FakeTable | None = None, metrics: Any = None) -> dict[str, float]:
        if target_table is None:
            target_table = self._tables["test"]
        target = target_table.df[self.target_col].to_numpy()
        if self.task_type.value == "binary_classification":
            predicted = (pred >= 0.5).astype(int)
            return {"accuracy": float((predicted == target).mean())}
        return {"mae": float(np.abs(pred.astype(float) - target.astype(float)).mean())}


def make_binary_task() -> FakeEntityTask:
    return FakeEntityTask(
        task_type=FakeTaskType.BINARY_CLASSIFICATION,
        train_targets=[0, 1, 0],
        val_targets=[1, 0],
        test_targets=[1, 0],
    )


def make_regression_task() -> FakeEntityTask:
    return FakeEntityTask(
        task_type=FakeTaskType.REGRESSION,
        train_targets=[1.0, 2.0, 3.0],
        val_targets=[2.0, 2.5],
        test_targets=[3.0, 4.0],
    )
