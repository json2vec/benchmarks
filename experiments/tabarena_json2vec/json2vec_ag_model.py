from __future__ import annotations

import io
from collections import Counter
from typing import Any

import lightning.pytorch as lit
import numpy as np
import pandas as pd
import torch
from autogluon.core.models import AbstractModel

import json2vec as j2v


def _records_from_frame(
    X: pd.DataFrame,
    y: pd.Series | None = None,
    feature_name_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    frame = X.copy()
    if feature_name_map is not None:
        frame = frame.rename(columns=feature_name_map)
    frame = frame.where(pd.notna(frame), None)
    records = frame.to_dict(orient="records")
    if y is not None:
        labels = y.tolist()
        for record, label in zip(records, labels, strict=True):
            record["__target__"] = None if pd.isna(label) else label
    return records


def _encoded_records_from_frame(
    X: pd.DataFrame,
    y: pd.Series | None = None,
    feature_name_map: dict[str, str] | None = None,
) -> list[list[dict[str, Any]]]:
    return [
        [record]
        for record in _records_from_frame(X=X, y=y, feature_name_map=feature_name_map)
    ]


def _safe_name(name: Any) -> str:
    return "".join(char if char.isalnum() or char == "_" else "_" for char in str(name))


def _validate_attention_shape(*, d_model: int, n_heads: int) -> None:
    if d_model % n_heads != 0:
        raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
    if d_model // n_heads < 2:
        raise ValueError(
            f"d_model={d_model} with n_heads={n_heads} gives head_dim={d_model // n_heads}; "
            "json2vec requires head_dim >= 2"
        )


def _resolve_accelerator(accelerator: str) -> str:
    if accelerator == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    return accelerator


def _field_for_column(
    name: str,
    series: pd.Series,
    *,
    max_vocab_size: int,
    p_mask: float | None = None,
) -> j2v.RequestBase:
    query = f"[*].{name}"
    kwargs = {"p_mask": p_mask} if p_mask is not None else {}
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return j2v.Number(name=_safe_name(name), query=query, **kwargs)
    return j2v.Category(name=_safe_name(name), query=query, max_vocab_size=max_vocab_size, **kwargs)


class JSON2VecTabArenaModel(AbstractModel):
    ag_key = "J2V"
    ag_name = "JSON2Vec"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: j2v.Architecture | None = None
        self.target_address: j2v.Address | None = None
        self.classes_: list[Any] | None = None
        self._feature_names: list[str] | None = None
        self._feature_name_map: dict[str, str] | None = None
        self._fallback_label: Any = None
        self._class_label_lookup: dict[str, Any] = {}
        self._target_mean: float = 0.0
        self._target_std: float = 1.0
        self._json2vec_payload: bytes | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        model = state.pop("model", None)
        if model is None:
            state["_json2vec_payload"] = None
            return state

        state_dict = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in model.state_dict().items()
        }
        payload = {
            "state_dict": state_dict,
            "hyperparameters": model.hyperparameters.model_dump(mode="python"),
            "batch_size": model.batch_size,
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        state["_json2vec_payload"] = buffer.getvalue()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        payload_bytes = state.pop("_json2vec_payload", None)
        self.__dict__.update(state)
        self.model = None
        self._json2vec_payload = payload_bytes

    def _ensure_model(self) -> j2v.Architecture:
        if self.model is not None:
            return self.model

        payload_bytes = self._json2vec_payload
        if payload_bytes is None:
            raise RuntimeError("model has not been fitted")

        payload = torch.load(io.BytesIO(payload_bytes), map_location="cpu", weights_only=False)
        hyperparameters = j2v.Hyperparameters.model_validate(payload["hyperparameters"])
        self.model = j2v.Architecture(
            hyperparameters=hyperparameters,
            batch_size=payload["batch_size"],
        )
        self.model.load_state_dict(payload["state_dict"], strict=False)
        return self.model

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        params = self._get_model_params()
        d_model = int(params.get("d_model", 32))
        batch_size = int(params.get("batch_size", 64))
        max_epochs = int(params.get("max_epochs", 8))
        lr = float(params.get("lr", 1e-3))
        weight_decay = float(params.get("weight_decay", 0.0))
        pretrain_epochs = int(params.get("pretrain_epochs", 0))
        pretrain_p_mask = float(params.get("pretrain_p_mask", 0.0))
        pretrain_lr = float(params.get("pretrain_lr", lr))
        max_cat_vocab_size = int(params.get("max_cat_vocab_size", 256))
        accelerator = str(params.get("accelerator", "mps"))
        attention = str(params.get("attention", "none"))
        dropout = float(params.get("dropout", 0.0))
        p_mask = float(params.get("p_mask", 0.0))
        p_target = float(params.get("p_target", 0.0))
        n_layers = int(params.get("n_layers", 1))
        n_linear = int(params.get("n_linear", 1))
        n_heads = int(params.get("n_heads", 4))
        random_seed = int(params.get("random_seed", 0))
        accelerator = _resolve_accelerator(accelerator)
        _validate_attention_shape(d_model=d_model, n_heads=n_heads)
        lit.seed_everything(random_seed, workers=True)

        self._feature_names = list(X.columns)
        self._feature_name_map = {
            column: f"f_{index}_{_safe_name(column)}"
            for index, column in enumerate(self._feature_names)
        }
        is_regression = self.problem_type == "regression"
        if not is_regression:
            self.classes_ = list(pd.Series(y).dropna().unique())
            try:
                self.classes_ = sorted(self.classes_)
            except TypeError:
                self.classes_ = sorted(self.classes_, key=str)
            self._class_label_lookup = {str(label): label for label in self.classes_}
            self._fallback_label = Counter(pd.Series(y).dropna().tolist()).most_common(1)[0][0]
            target_vocab_size = max(len(self.classes_) + 2, 8)
            target_field = j2v.Category(
                name="__target__",
                query="[*].__target__",
                max_vocab_size=target_vocab_size,
                topk=[min(len(self.classes_), target_vocab_size - 1)],
            )
        else:
            target_field = j2v.Number(name="__target__", query="[*].__target__")
            target = pd.Series(y, dtype="float64")
            self._target_mean = float(target.mean())
            target_std = float(target.std(ddof=0))
            self._target_std = target_std if np.isfinite(target_std) and target_std > 1e-12 else 1.0

        fields = [
            _field_for_column(
                name=self._feature_name_map[column],
                series=X[column],
                max_vocab_size=max_cat_vocab_size,
            )
            for column in self._feature_names
        ]

        self.target_address = j2v.Address("record", "__target__")

        @j2v.shim(yields=True)
        def tabarena_records(observation: dict, strata: j2v.Strata, records: list[dict[str, Any]]):
            yield from records

        def make_trainer(epochs: int) -> lit.Trainer:
            return lit.Trainer(
                accelerator=accelerator,
                devices=1,
                max_epochs=epochs,
                logger=False,
                enable_checkpointing=False,
                enable_model_summary=False,
                enable_progress_bar=False,
                limit_val_batches=0,
                num_sanity_val_steps=0,
            )

        def make_datamodule(model: j2v.Architecture, records: list[dict[str, Any]]) -> Any:
            dataset = j2v.Dataset(
                processor=tabarena_records,
                kwargs={"records": records},
            )
            return j2v.StreamingDataModule.from_model(
                model,
                dataset=dataset,
                num_workers=0,
                persistent_workers=False,
                pin_memory=False,
                file_buffer_size=1,
                observation_buffer_size=max(batch_size * 2, 32),
                sample_rate=1.0,
            )

        pretrained_state_dict: dict[str, Any] | None = None
        if pretrain_epochs > 0 and pretrain_p_mask > 0.0:
            pretrain_fields = [
                _field_for_column(
                    name=self._feature_name_map[column],
                    series=X[column],
                    max_vocab_size=max_cat_vocab_size,
                    p_mask=pretrain_p_mask,
                )
                for column in self._feature_names
            ]
            pretrain_hyperparameters = j2v.Hyperparameters(
                d_model=d_model,
                fields=j2v.Array(
                    name="record",
                    fields=pretrain_fields,
                    max_length=1,
                    n_outputs=1,
                    attention=attention,
                    dropout=dropout,
                    n_layers=n_layers,
                    n_linear=n_linear,
                    n_heads=n_heads,
                ),
                target=[],
            )
            pretrain_model = j2v.Architecture(
                hyperparameters=pretrain_hyperparameters,
                batch_size=batch_size,
                optimizer=lambda module: torch.optim.AdamW(
                    module.parameters(),
                    lr=pretrain_lr,
                    weight_decay=weight_decay,
                ),
            )
            pretrain_datamodule = make_datamodule(
                pretrain_model,
                _records_from_frame(X, feature_name_map=self._feature_name_map),
            )
            make_trainer(pretrain_epochs).fit(model=pretrain_model, datamodule=pretrain_datamodule)
            pretrained_state_dict = {
                key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                for key, value in pretrain_model.state_dict().items()
            }

        fields.append(target_field)

        hyperparameters = j2v.Hyperparameters(
            d_model=d_model,
            fields=j2v.Array(
                name="record",
                fields=fields,
                max_length=1,
                n_outputs=1,
                attention=attention,
                dropout=dropout,
                p_mask=p_mask,
                p_target=p_target,
                n_layers=n_layers,
                n_linear=n_linear,
                n_heads=n_heads,
            ),
            target=self.target_address,
        )
        self.model = j2v.Architecture(
            hyperparameters=hyperparameters,
            batch_size=batch_size,
            optimizer=lambda module: torch.optim.AdamW(
                module.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            ),
        )

        if pretrained_state_dict is not None:
            current_state = self.model.state_dict()
            for key, value in pretrained_state_dict.items():
                if (
                    key in current_state
                    and isinstance(current_state[key], torch.Tensor)
                    and isinstance(value, torch.Tensor)
                    and current_state[key].shape == value.shape
                ):
                    current_state[key] = value
            self.model.load_state_dict(current_state, strict=True)

        datamodule = make_datamodule(
            self.model,
            _records_from_frame(
                X,
                self._scale_target(y) if is_regression else y,
                feature_name_map=self._feature_name_map,
            ),
        )
        make_trainer(max_epochs).fit(model=self.model, datamodule=datamodule)
        return self

    def _scale_target(self, y: pd.Series) -> pd.Series:
        return (pd.Series(y, dtype="float64") - self._target_mean) / self._target_std

    def _unscale_target(self, y: np.ndarray) -> np.ndarray:
        return y.astype(np.float32) * np.float32(self._target_std) + np.float32(self._target_mean)

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame | np.ndarray:
        if self.target_address is None:
            raise RuntimeError("model has not been fitted")

        model = self._ensure_model()
        outputs = model.predict(
            _encoded_records_from_frame(X[self._feature_names], feature_name_map=self._feature_name_map)
        )
        content = outputs[self.target_address]["content"]

        if self.problem_type == "regression":
            return self._unscale_target(np.asarray(content, dtype=np.float32))

        if self.classes_ is None:
            raise RuntimeError("classification labels were not initialized")
        topk = content["topk"]

        result = pd.DataFrame(0.0, columns=self.classes_, index=X.index, dtype=np.float32)
        for idx, candidates in enumerate(topk):
            for candidate in candidates:
                label = self._class_label_lookup.get(str(candidate["label"]), candidate["label"])
                if label in result.columns:
                    result.iat[idx, result.columns.get_loc(label)] = float(candidate["probability"])

        row_sums = result.sum(axis=1)
        for row_idx, row_sum in row_sums.items():
            if row_sum <= 0:
                result.loc[row_idx, self._fallback_label] = 1.0
            elif abs(row_sum - 1.0) > 1e-6:
                result.loc[row_idx] = result.loc[row_idx] / row_sum

        if self.problem_type == "binary":
            return result[self.classes_[1]].to_numpy(dtype=np.float32)
        return result

    def _predict(self, X: pd.DataFrame, **kwargs) -> pd.Series:
        proba = self._predict_proba(X, **kwargs)
        if self.problem_type == "regression":
            return pd.Series(proba, index=X.index)
        if isinstance(proba, pd.DataFrame):
            return proba.idxmax(axis=1)
        return pd.Series(np.where(proba >= 0.5, self.classes_[1], self.classes_[0]), index=X.index)

    def _set_default_params(self):
        defaults = {
            "d_model": 32,
            "batch_size": 64,
            "max_epochs": 8,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "pretrain_epochs": 0,
            "pretrain_p_mask": 0.0,
            "pretrain_lr": 1e-3,
            "max_cat_vocab_size": 256,
            "accelerator": "mps",
            "attention": "none",
            "dropout": 0.0,
            "p_mask": 0.0,
            "p_target": 0.0,
            "n_layers": 1,
            "n_linear": 1,
            "n_heads": 4,
            "random_seed": 0,
        }
        for key, value in defaults.items():
            self._set_default_param_value(key, value)

    def _get_default_auxiliary_params(self) -> dict:
        params = super()._get_default_auxiliary_params()
        params.update({"valid_raw_types": ["int", "float", "category", "bool", "object"]})
        return params
