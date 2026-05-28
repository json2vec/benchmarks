from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(BENCHMARKS_DIR))
sys.path.insert(0, str(EXPERIMENT_DIR))

from relbench_json2vec.config import JSON2VecConfig
from relbench_json2vec.paths import EXPERIMENT_DIR as DEFAULT_EXPERIMENT_DIR
from relbench_json2vec.sdk import BenchmarkReport, BenchmarkRun, BenchmarkSpec, inspect_task


def model_config_from_args(args: argparse.Namespace) -> JSON2VecConfig:
    return JSON2VecConfig(
        d_model=args.d_model,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        accelerator=args.accelerator,
        attention=args.attention,
        n_layers=args.n_layers,
        n_linear=args.n_linear,
        n_heads=args.n_heads,
        random_seed=args.seed,
        history_limit=args.history_limit,
        relation_depth=args.relation_depth,
        max_cat_vocab_size=args.max_cat_vocab_size,
    )


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--accelerator", default="mps")
    parser.add_argument("--attention", default="mha")
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-linear", type=int, default=1)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history-limit", type=int, default=128)
    parser.add_argument("--relation-depth", type=int, default=1)
    parser.add_argument("--max-cat-vocab-size", type=int, default=2048)


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tasks", default="rel-f1:driver-dnf")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR / "relbench_cache")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_EXPERIMENT_DIR / "runs")
    parser.add_argument("--notes", default="single-task RelBench quality run")
    parser.add_argument("--scope", default="relbench_json2vec_single_task")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--download", dest="download", action="store_true", default=True)
    parser.add_argument("--no-download", dest="download", action="store_false")


def command_run(args: argparse.Namespace) -> None:
    spec = BenchmarkSpec.from_task_string(
        args.tasks,
        model_config=model_config_from_args(args),
        cache_dir=args.cache_dir,
        output_root=args.output_root,
        notes=args.notes,
        seed=args.seed,
        download=args.download,
        scope=args.scope,
        append_results=args.append_results,
        append_results_csv=args.append_results_csv,
    )
    run = BenchmarkRun(spec, run_id=args.run_id)
    if args.dry_run:
        print(json.dumps(run.dry_run(), indent=2, sort_keys=True))
        return
    report_path = run.run()
    print(report_path)


def command_report(args: argparse.Namespace) -> None:
    report = BenchmarkReport.from_csv(args.results)
    if args.output is not None:
        report.write_markdown(args.output)
        print(args.output)
    else:
        print(report.to_markdown())


def _fake_task(kind: str):
    from tests.fakes import make_binary_task, make_regression_task

    if kind == "binary":
        return make_binary_task()
    if kind == "regression":
        return make_regression_task()
    raise ValueError(f"unsupported fake task: {kind}")


def command_inspect(args: argparse.Namespace) -> None:
    if args.fake is not None:
        task = _fake_task(args.fake)
        dataset_name = "fake"
        task_name = args.fake
    else:
        task = None
        dataset_name, task_name = args.tasks.split(":", 1)

    summary = inspect_task(
        dataset_name=dataset_name,
        task_name=task_name,
        download=args.download,
        cache_dir=args.cache_dir,
        history_limit=args.history_limit,
        relation_depth=args.relation_depth,
        limit=args.limit,
        task=task,
    )
    print(json.dumps(summary, indent=2, default=str, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RelBench json2vec benchmark SDK CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run json2vec and simple baselines")
    add_common_model_args(run_parser)
    add_runtime_args(run_parser)
    run_parser.add_argument("--append-results", action="store_true", default=False)
    run_parser.add_argument("--append-results-csv", type=Path, default=DEFAULT_EXPERIMENT_DIR / "results.csv")
    run_parser.add_argument("--dry-run", action="store_true", default=False)
    run_parser.set_defaults(func=command_run)

    report_parser = subparsers.add_parser("report", help="Summarize a results CSV")
    report_parser.add_argument("--results", type=Path, required=True)
    report_parser.add_argument("--output", type=Path, default=None)
    report_parser.set_defaults(func=command_report)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect materialized records and inferred schema")
    inspect_parser.add_argument("--tasks", default="rel-f1:driver-dnf")
    inspect_parser.add_argument("--cache-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR / "relbench_cache")
    inspect_parser.add_argument("--download", dest="download", action="store_true", default=True)
    inspect_parser.add_argument("--no-download", dest="download", action="store_false")
    inspect_parser.add_argument("--history-limit", type=int, default=128)
    inspect_parser.add_argument("--relation-depth", type=int, default=1)
    inspect_parser.add_argument("--limit", type=int, default=2)
    inspect_parser.add_argument("--fake", choices=("binary", "regression"), default=None)
    inspect_parser.set_defaults(func=command_inspect)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
