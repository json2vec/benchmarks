from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
import torch

from relbench_json2vec.config import JSON2VecConfig
from relbench_json2vec.materialize import MaterializedSplit, materialize_task
from relbench_json2vec.schema import build_schema


@dataclass
class JSON2VecTaskResult:
    dataset: str
    task: str
    task_type: str
    model_or_config: str
    metrics_by_split: dict[str, dict[str, float]]
    time_train_s: float
    time_infer_s: dict[str, float]
    warnings: list[str] = field(default_factory=list)


def _resolve_accelerator(accelerator: str) -> str:
    if accelerator == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if accelerator == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return accelerator


def _records_frame(records: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame({"payload": pl.Series("payload", records, dtype=pl.Object)})


def _payload_preprocessor(row: dict[str, Any]) -> dict[str, Any]:
    payload = row["payload"]
    if not isinstance(payload, dict):
        raise TypeError(f"payload must be a dict, got {type(payload).__name__}")
    return payload


def _task_type_value(task: Any) -> str:
    return getattr(task.task_type, "value", str(task.task_type))


def _trainer(config: JSON2VecConfig) -> lit.Trainer:
    import lightning.pytorch as lit

    return lit.Trainer(
        accelerator=_resolve_accelerator(config.accelerator),
        devices=1,
        max_epochs=config.max_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )


def _datamodule(j2v: Any, model: Any, train: MaterializedSplit, val: MaterializedSplit) -> Any:
    return j2v.PolarsDataModule(
        model=model,
        train=_records_frame(train.records),
        validate=_records_frame(val.records),
        preprocessor=_payload_preprocessor,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        observation_buffer_size=256,
        sample_rate=1.0,
    )


def _positive_probability(outputs: dict[str, Any], *, classes: list[Any]) -> np.ndarray:
    topk_rows = outputs["content"]["topk"]
    positive = classes[-1]
    probabilities: list[float] = []
    for candidates in topk_rows:
        row = {str(candidate["label"]): float(candidate["probability"]) for candidate in candidates}
        probability = row.get(str(positive), 0.0)
        total = sum(row.values())
        if total > 0:
            probability /= total
        probabilities.append(probability)
    return np.asarray(probabilities, dtype=np.float32)


def _predictions_for_task(
    *,
    model: Any,
    target_address: Any,
    task: Any,
    records: list[dict[str, Any]],
    classes: list[Any],
) -> np.ndarray:
    outputs = model.predict(records)
    target_outputs = outputs[target_address]
    task_type = _task_type_value(task)
    if task_type == "regression":
        return np.asarray(target_outputs["content"], dtype=np.float32)
    if task_type == "binary_classification":
        return _positive_probability(target_outputs, classes=classes)
    raise ValueError(f"unsupported RelBench task type for v1: {task_type}")


def fit_and_evaluate_json2vec(
    *,
    j2v: Any,
    dataset_name: str,
    task_name: str,
    task: Any,
    config: JSON2VecConfig,
) -> JSON2VecTaskResult:
    import lightning.pytorch as lit

    lit.seed_everything(config.random_seed, workers=True)
    materialized = materialize_task(
        dataset_name=dataset_name,
        task_name=task_name,
        task=task,
        history_limit=config.history_limit,
        relation_depth=config.relation_depth,
    )
    train_records = materialized["train"].records
    val_records = materialized["val"].records
    schema = build_schema(
        j2v=j2v,
        records=train_records + val_records,
        task_type=task.task_type,
        history_limit=config.history_limit,
        max_cat_vocab_size=config.max_cat_vocab_size,
    )
    model = j2v.Model.from_schema(
        *schema.fields,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        batch_size=config.batch_size,
        attention=config.attention,
        n_linear=config.n_linear,
        optimizer=lambda module: torch.optim.AdamW(
            module.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        ),
    )

    train_start = time.time()
    _trainer(config).fit(
        model=model,
        datamodule=_datamodule(j2v, model, materialized["train"], materialized["val"]),
    )
    time_train_s = time.time() - train_start

    classes: list[Any] = []
    if _task_type_value(task) == "binary_classification":
        classes = sorted(set(task.get_table("train", mask_input_cols=False).df[task.target_col].dropna().tolist()))

    metrics_by_split: dict[str, dict[str, float]] = {}
    time_infer_s: dict[str, float] = {}
    for split_name in ("val", "test"):
        split = materialized[split_name]
        infer_start = time.time()
        pred = _predictions_for_task(
            model=model,
            target_address=schema.target_address,
            task=task,
            records=split.records,
            classes=classes,
        )
        time_infer_s[split_name] = time.time() - infer_start
        target_table = None if split_name == "test" else task.get_table(split_name, mask_input_cols=False)
        metrics_by_split[split_name] = {
            name: float(value)
            for name, value in task.evaluate(pred, target_table=target_table).items()
        }

    warnings = sorted(
        set(
            schema.warnings
            + materialized["train"].warnings
            + materialized["val"].warnings
            + materialized["test"].warnings
        )
    )
    return JSON2VecTaskResult(
        dataset=dataset_name,
        task=task_name,
        task_type=_task_type_value(task),
        model_or_config=config.label(),
        metrics_by_split=metrics_by_split,
        time_train_s=time_train_s,
        time_infer_s=time_infer_s,
        warnings=warnings,
    )
