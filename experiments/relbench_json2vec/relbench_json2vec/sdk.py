from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from relbench_json2vec.baselines import BaselineTaskResult
from relbench_json2vec.config import JSON2VecConfig, RelBenchRunConfig, parse_task_specs
from relbench_json2vec.materialize import materialize_task
from relbench_json2vec.paths import EXPERIMENT_DIR, REPO_ROOT, configure_import_paths
from relbench_json2vec.results import FIELDNAMES, append_results_csv
from relbench_json2vec.runner import _load_task, _rows_from_result, run_baselines_from_env, run_json2vec_from_env
from relbench_json2vec.schema import build_schema


HIGHER_IS_BETTER = {
    "accuracy",
    "average_precision",
    "f1",
    "link_prediction_map",
    "link_prediction_precision",
    "link_prediction_recall",
    "macro_f1",
    "micro_f1",
    "mrr",
    "r2",
    "roc_auc",
}
LOWER_IS_BETTER = {"mae", "rmse", "log_loss", "mse"}


@dataclass(frozen=True)
class BenchmarkSpec:
    tasks: tuple[tuple[str, str], ...]
    model_config: JSON2VecConfig
    models: tuple[str, ...] = ("baselines", "json2vec")
    cache_dir: Path = EXPERIMENT_DIR / "relbench_cache"
    output_root: Path = EXPERIMENT_DIR / "runs"
    notes: str = ""
    seed: int = 0
    download: bool = True
    scope: str = "relbench_json2vec_single_task"
    append_results: bool = False
    append_results_csv: Path = EXPERIMENT_DIR / "results.csv"

    @classmethod
    def single_task_default(cls) -> "BenchmarkSpec":
        return cls(
            tasks=(("rel-f1", "driver-dnf"),),
            model_config=JSON2VecConfig(
                d_model=32,
                batch_size=64,
                max_epochs=3,
                lr=0.001,
                weight_decay=0.0,
                accelerator="mps",
                attention="mha",
                n_layers=1,
                n_linear=1,
                n_heads=4,
                random_seed=0,
                history_limit=128,
                relation_depth=1,
                max_cat_vocab_size=2048,
            ),
            notes="single-task RelBench quality run",
        )

    @classmethod
    def from_task_string(
        cls,
        raw_tasks: str,
        *,
        model_config: JSON2VecConfig,
        cache_dir: Path,
        output_root: Path,
        notes: str,
        seed: int,
        download: bool,
        scope: str,
        append_results: bool,
        append_results_csv: Path,
        models: tuple[str, ...] = ("baselines", "json2vec"),
    ) -> "BenchmarkSpec":
        return cls(
            tasks=parse_task_specs(raw_tasks),
            model_config=model_config,
            models=models,
            cache_dir=cache_dir,
            output_root=output_root,
            notes=notes,
            seed=seed,
            download=download,
            scope=scope,
            append_results=append_results,
            append_results_csv=append_results_csv,
        )

    def to_manifest(self, *, run_id: str, run_dir: Path, status: str) -> dict[str, Any]:
        payload = asdict(self)
        payload["tasks"] = [f"{dataset}:{task}" for dataset, task in self.tasks]
        payload["cache_dir"] = str(self.cache_dir)
        payload["output_root"] = str(self.output_root)
        payload["append_results_csv"] = str(self.append_results_csv)
        payload["run_id"] = run_id
        payload["run_dir"] = str(run_dir)
        payload["status"] = status
        payload["git"] = git_info()
        payload["packages"] = package_versions()
        payload["command"] = sys.argv
        return payload


def task_slug(tasks: tuple[tuple[str, str], ...]) -> str:
    if len(tasks) == 1:
        dataset, task = tasks[0]
        return f"{dataset}-{task}".replace(":", "-")
    return f"{len(tasks)}-tasks"


def default_run_id(tasks: tuple[tuple[str, str], ...], *, now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    return f"{current.strftime('%Y%m%d_%H%M%S')}-{task_slug(tasks)}"


def git_info() -> dict[str, str | bool | None]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = run(["git", "status", "--short"])
    return {
        "repo": str(REPO_ROOT),
        "sha": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(status),
    }


def package_versions() -> dict[str, str]:
    packages: dict[str, str] = {}
    for package in ("json2vec", "relbench", "lightning", "torch", "torchmetrics", "pandas", "polars"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"
    return packages


class BenchmarkRun:
    def __init__(self, spec: BenchmarkSpec, *, run_id: str | None = None):
        self.spec = spec
        self.run_id = run_id or default_run_id(spec.tasks)
        self.run_dir = spec.output_root / self.run_id
        self.results_csv = self.run_dir / "results.csv"
        self.manifest_path = self.run_dir / "manifest.json"
        self.report_path = self.run_dir / "report.md"

    def write_manifest(self, *, status: str, error: str | None = None) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.spec.to_manifest(run_id=self.run_id, run_dir=self.run_dir, status=status)
        if error is not None:
            manifest["error"] = error
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    def dry_run(self) -> dict[str, Any]:
        self.write_manifest(status="dry-run")
        return self.spec.to_manifest(run_id=self.run_id, run_dir=self.run_dir, status="dry-run")

    def run(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.write_manifest(status="running")
        run_config = RelBenchRunConfig(
            tasks=self.spec.tasks,
            download=self.spec.download,
            cache_dir=self.spec.cache_dir,
            results_csv=self.results_csv,
            results_scope=self.spec.scope,
            results_notes=self.spec.notes,
        )
        try:
            if "baselines" in self.spec.models:
                with (self.run_dir / "baselines.log").open("w") as stdout, (
                    self.run_dir / "baselines.err"
                ).open("w") as stderr:
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        run_baselines_from_env(run_config)
            if "json2vec" in self.spec.models:
                with (self.run_dir / "json2vec.log").open("w") as stdout, (
                    self.run_dir / "json2vec.err"
                ).open("w") as stderr:
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        run_json2vec_from_env(run_config, self.spec.model_config)

            report = BenchmarkReport.from_csv(self.results_csv)
            report.write_markdown(self.report_path)
            if self.spec.append_results:
                append_results_csv(self.spec.append_results_csv, read_result_rows(self.results_csv))
            self.write_manifest(status="complete")
            return self.report_path
        except Exception as error:
            self.write_manifest(status="failed", error=f"{type(error).__name__}: {error}")
            raise


def read_result_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def metric_direction(metric: str) -> int:
    normalized = metric.lower()
    if normalized in LOWER_IS_BETTER:
        return -1
    return 1


def _score(value: float, metric: str) -> float:
    return metric_direction(metric) * value


@dataclass(frozen=True)
class ReportComparison:
    dataset: str
    task: str
    split: str
    metric: str
    json2vec_model: str
    json2vec_value: float
    baseline_model: str
    baseline_value: float

    @property
    def delta(self) -> float:
        direction = metric_direction(self.metric)
        return direction * (self.json2vec_value - self.baseline_value)

    @property
    def verdict(self) -> str:
        if abs(self.delta) < 1e-12:
            return "tie"
        return "better" if self.delta > 0 else "worse"


@dataclass
class BenchmarkReport:
    rows: list[dict[str, Any]]
    comparisons: list[ReportComparison]

    @classmethod
    def from_csv(cls, path: Path) -> "BenchmarkReport":
        rows = read_result_rows(path)
        return cls(rows=rows, comparisons=build_comparisons(rows))

    def summary_lines(self) -> list[str]:
        lines = ["# RelBench json2vec Benchmark Report", ""]
        if not self.rows:
            return lines + ["No result rows found."]

        wins = sum(1 for comparison in self.comparisons if comparison.verdict == "better")
        losses = sum(1 for comparison in self.comparisons if comparison.verdict == "worse")
        ties = sum(1 for comparison in self.comparisons if comparison.verdict == "tie")
        lines.append(f"Comparisons against best simple baseline: {wins} better, {losses} worse, {ties} tied.")
        lines.append("")
        lines.append("| dataset | task | split | metric | json2vec | best baseline | delta | verdict |")
        lines.append("|---|---|---|---|---:|---:|---:|---|")
        for comparison in self.comparisons:
            lines.append(
                "| "
                f"{comparison.dataset} | {comparison.task} | {comparison.split} | {comparison.metric} | "
                f"{comparison.json2vec_value:.6g} ({comparison.json2vec_model}) | "
                f"{comparison.baseline_value:.6g} ({comparison.baseline_model}) | "
                f"{comparison.delta:.6g} | {comparison.verdict} |"
            )
        return lines

    def to_markdown(self) -> str:
        return "\n".join(self.summary_lines()) + "\n"

    def write_markdown(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown())


def build_comparisons(rows: list[dict[str, Any]]) -> list[ReportComparison]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["dataset"], row["task"], row["split"], row["metric"])
        groups.setdefault(key, []).append(row)

    comparisons: list[ReportComparison] = []
    for (dataset, task, split, metric), group in sorted(groups.items()):
        json2vec_rows = [row for row in group if row["model_or_config"].startswith("json2vec")]
        baseline_rows = [row for row in group if not row["model_or_config"].startswith("json2vec")]
        if not json2vec_rows or not baseline_rows:
            continue
        best_json2vec = max(json2vec_rows, key=lambda row: _score(float(row["value"]), metric))
        best_baseline = max(baseline_rows, key=lambda row: _score(float(row["value"]), metric))
        comparisons.append(
            ReportComparison(
                dataset=dataset,
                task=task,
                split=split,
                metric=metric,
                json2vec_model=best_json2vec["model_or_config"],
                json2vec_value=float(best_json2vec["value"]),
                baseline_model=best_baseline["model_or_config"],
                baseline_value=float(best_baseline["value"]),
            )
        )
    return comparisons


@dataclass
class ProbeField:
    name: str
    type: str
    fields: list[Any] = field(default_factory=list)


class SchemaProbe:
    @staticmethod
    def Category(name: str, **kwargs: Any) -> ProbeField:
        return ProbeField(name=name, type="category")

    @staticmethod
    def Number(name: str, **kwargs: Any) -> ProbeField:
        return ProbeField(name=name, type="number")

    @staticmethod
    def DateParts(name: str, **kwargs: Any) -> ProbeField:
        return ProbeField(name=name, type="dateparts")

    @staticmethod
    def Set(name: str, **kwargs: Any) -> ProbeField:
        return ProbeField(name=name, type="set")

    @staticmethod
    def Array(*children: Any, name: str, **kwargs: Any) -> ProbeField:
        return ProbeField(name=name, type="array", fields=list(children))

    @staticmethod
    def Address(*parts: str) -> str:
        return "/".join(parts)


def inspect_task(
    *,
    dataset_name: str,
    task_name: str,
    download: bool,
    cache_dir: Path,
    history_limit: int,
    relation_depth: int,
    limit: int,
    task: Any | None = None,
) -> dict[str, Any]:
    if task is None:
        task = _load_task(dataset_name, task_name, download=download, cache_dir=str(cache_dir / dataset_name / task_name))
    materialized = materialize_task(
        dataset_name=dataset_name,
        task_name=task_name,
        task=task,
        history_limit=history_limit,
        relation_depth=relation_depth,
    )
    train_records = materialized["train"].records
    val_records = materialized["val"].records
    schema = build_schema(
        j2v=SchemaProbe,
        records=train_records + val_records,
        task_type=task.task_type,
        history_limit=history_limit,
        max_cat_vocab_size=2048,
    )
    return {
        "dataset": dataset_name,
        "task": task_name,
        "splits": {split: len(value.records) for split, value in materialized.items()},
        "sample_records": materialized["train"].records[:limit],
        "schema_fields": summarize_fields(schema.fields),
        "warnings": sorted(
            set(
                schema.warnings
                + materialized["train"].warnings
                + materialized["val"].warnings
                + materialized["test"].warnings
            )
        ),
    }


def summarize_fields(fields: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for field_obj in fields:
        children = getattr(field_obj, "fields", [])
        out.append(
            {
                "name": getattr(field_obj, "name", ""),
                "type": getattr(field_obj, "type", ""),
                "children": summarize_fields(children) if children else [],
            }
        )
    return out
