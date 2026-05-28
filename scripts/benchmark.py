from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


BENCHMARKS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARKS_ROOT))

from benchmark_common.cli import add_bool_pair, jsonable
from benchmark_common.env import overlay_env
from benchmark_common.paths import (
    REL_BENCH_EXPERIMENT_DIR,
    TABARENA_EXPERIMENT_DIR,
    configure_relbench_paths,
    configure_tabarena_paths,
)


def _add_json2vec_core_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--attention", default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--n-linear", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-cat-vocab-size", type=int, default=None)


def _json2vec_env_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "JSON2VEC_D_MODEL": args.d_model,
        "JSON2VEC_BATCH_SIZE": args.batch_size,
        "JSON2VEC_MAX_EPOCHS": args.max_epochs,
        "JSON2VEC_LR": args.lr,
        "JSON2VEC_WEIGHT_DECAY": args.weight_decay,
        "JSON2VEC_ACCELERATOR": args.accelerator,
        "JSON2VEC_ATTENTION": args.attention,
        "JSON2VEC_N_LAYERS": args.n_layers,
        "JSON2VEC_N_LINEAR": args.n_linear,
        "JSON2VEC_N_HEADS": args.n_heads,
        "JSON2VEC_RANDOM_SEED": args.seed,
        "JSON2VEC_MAX_CAT_VOCAB_SIZE": args.max_cat_vocab_size,
    }


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(jsonable(payload), indent=2, sort_keys=True))


def _tabarena_env(args: argparse.Namespace) -> dict[str, str]:
    overrides = {
        "TABARENA_TASK_IDS": args.task_ids,
        "TABARENA_MAX_INSTANCES": args.max_instances,
        "TABARENA_TASK_LIMIT": args.task_limit,
        "TABARENA_CACHE_MODE": args.cache_mode,
        "TABARENA_REPETITIONS_MODE": args.repetitions_mode,
        "TABARENA_OUTPUT_DIR": args.output_dir,
        "TABARENA_RESULTS_CSV": args.results_csv,
        "TABARENA_RESULTS_SCOPE": args.scope,
        "TABARENA_RESULTS_NOTES": args.notes,
        "TABARENA_DEBUG_MODE": args.debug_mode,
        "TABARENA_RAISE_ON_FAILURE": args.raise_on_failure,
        **_json2vec_env_overrides(args),
        "JSON2VEC_PRETRAIN_EPOCHS": args.pretrain_epochs,
        "JSON2VEC_PRETRAIN_P_MASK": args.pretrain_p_mask,
        "JSON2VEC_PRETRAIN_LR": args.pretrain_lr,
        "JSON2VEC_DROPOUT": args.dropout,
        "JSON2VEC_P_MASK": args.p_mask,
        "JSON2VEC_P_TARGET": args.p_target,
        "JSON2VEC_NUM_BAG_FOLDS": args.num_bag_folds,
        "JSON2VEC_HOLDOUT": args.holdout,
        "RF_NUM_BAG_FOLDS": args.rf_num_bag_folds,
        "RF_NUM_RANDOM_CONFIGS": args.rf_num_random_configs,
    }
    return overlay_env(overrides)


def command_tabarena_run(args: argparse.Namespace) -> None:
    configure_tabarena_paths()
    from tabarena_json2vec.config import JSON2VecConfig, RandomForestConfig, TabArenaRunConfig, default_tabarena_output_dir

    env = _tabarena_env(args)
    if args.model == "json2vec":
        default_output_dir = default_tabarena_output_dir("tabarena_out")
        default_scope = "json2vec"
        model_config = JSON2VecConfig.from_env(env)
    else:
        default_output_dir = default_tabarena_output_dir("tabarena_rf_out")
        default_scope = "random_forest"
        model_config = RandomForestConfig.from_env(env)

    run_config = TabArenaRunConfig.from_env(
        default_output_dir=default_output_dir,
        default_results_scope=default_scope,
        env=env,
    )
    if args.dry_run:
        _print_json(
            {
                "suite": "tabarena",
                "command": "run",
                "status": "dry-run",
                "model": args.model,
                "run_config": run_config,
                "model_config": model_config,
            }
        )
        return

    from tabarena_json2vec.paths import configure_import_paths

    configure_import_paths()
    from tabarena_json2vec.runner import run_json2vec_from_env, run_random_forest_from_env

    if args.model == "json2vec":
        run_json2vec_from_env(run_config=run_config, model_config=model_config)
    else:
        run_random_forest_from_env(run_config=run_config, model_config=model_config)


def _relbench_env(args: argparse.Namespace) -> dict[str, str]:
    overrides = {
        "RELBENCH_TASKS": args.tasks,
        "RELBENCH_DOWNLOAD": args.download,
        "RELBENCH_CACHE_DIR": args.cache_dir,
        "RELBENCH_RESULTS_CSV": args.results_csv,
        "RELBENCH_RESULTS_SCOPE": args.scope,
        "RELBENCH_RESULTS_NOTES": args.notes,
        **_json2vec_env_overrides(args),
        "JSON2VEC_HISTORY_LIMIT": args.history_limit,
        "JSON2VEC_RELATION_DEPTH": args.relation_depth,
    }
    return overlay_env(overrides)


def _relbench_model_config_from_args(args: argparse.Namespace):
    from relbench_json2vec.config import JSON2VecConfig

    return JSON2VecConfig.from_env(_relbench_env(args))


def command_relbench_run(args: argparse.Namespace) -> None:
    configure_relbench_paths()
    from relbench_json2vec.config import RelBenchRunConfig
    from relbench_json2vec.paths import EXPERIMENT_DIR
    from relbench_json2vec.sdk import BenchmarkRun, BenchmarkSpec

    env = _relbench_env(args)
    default_scope = "relbench_json2vec_single_task" if args.model == "json2vec" else "relbench_simple_baselines"
    run_config = RelBenchRunConfig.from_env(default_results_scope=default_scope, env=env)
    model_config = _relbench_model_config_from_args(args)
    output_root = args.output_root or (EXPERIMENT_DIR / "runs")
    append_results_csv = Path(args.append_results_csv or run_config.results_csv or EXPERIMENT_DIR / "results.csv")
    spec = BenchmarkSpec(
        tasks=run_config.tasks,
        model_config=model_config,
        cache_dir=run_config.cache_dir,
        output_root=output_root,
        notes=run_config.results_notes,
        seed=model_config.random_seed,
        download=run_config.download,
        scope=run_config.results_scope,
        append_results=args.append_results,
        append_results_csv=append_results_csv,
        models=(args.model,),
    )
    run = BenchmarkRun(spec, run_id=args.run_id)
    if args.dry_run:
        _print_json(run.dry_run())
        return
    print(run.run())


def _fake_relbench_task(kind: str):
    from tests.fakes import make_binary_task, make_regression_task

    if kind == "binary":
        return make_binary_task()
    if kind == "regression":
        return make_regression_task()
    raise ValueError(f"unsupported fake task: {kind}")


def command_relbench_inspect(args: argparse.Namespace) -> None:
    configure_relbench_paths()
    from relbench_json2vec.sdk import inspect_task

    if args.fake is not None:
        task = _fake_relbench_task(args.fake)
        dataset_name = "fake"
        task_name = args.fake
    else:
        task = None
        dataset_name, task_name = args.tasks.split(":", 1)

    summary = inspect_task(
        dataset_name=dataset_name,
        task_name=task_name,
        download=args.download if args.download is not None else True,
        cache_dir=args.cache_dir or (REL_BENCH_EXPERIMENT_DIR / "relbench_cache"),
        history_limit=args.history_limit or 128,
        relation_depth=args.relation_depth or 1,
        limit=args.limit,
        task=task,
    )
    print(json.dumps(summary, indent=2, default=str, sort_keys=True))


def command_relbench_report(args: argparse.Namespace) -> None:
    configure_relbench_paths()
    from relbench_json2vec.sdk import BenchmarkReport

    report = BenchmarkReport.from_csv(args.results)
    if args.output is not None:
        report.write_markdown(args.output)
        print(args.output)
    else:
        print(report.to_markdown())


def add_tabarena_run_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("run", help="Run a TabArena benchmark model")
    parser.add_argument("--model", choices=("json2vec", "random-forest"), required=True)
    parser.add_argument("--task-ids", default=None)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--cache-mode", default=None)
    parser.add_argument("--repetitions-mode", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--scope", default=None)
    parser.add_argument("--notes", default=None)
    add_bool_pair(parser, "debug-mode", dest="debug_mode")
    add_bool_pair(parser, "raise-on-failure", dest="raise_on_failure")
    _add_json2vec_core_args(parser)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--pretrain-p-mask", type=float, default=None)
    parser.add_argument("--pretrain-lr", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--p-mask", type=float, default=None)
    parser.add_argument("--p-target", type=float, default=None)
    parser.add_argument("--num-bag-folds", type=int, default=None)
    add_bool_pair(parser, "holdout", dest="holdout")
    parser.add_argument("--rf-num-bag-folds", type=int, default=None)
    parser.add_argument("--rf-num-random-configs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.set_defaults(func=command_tabarena_run)


def add_relbench_run_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("run", help="Run a RelBench benchmark model")
    parser.add_argument("--model", choices=("json2vec", "baselines"), required=True)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--scope", default=None)
    parser.add_argument("--run-id", default=None)
    add_bool_pair(parser, "download", dest="download")
    _add_json2vec_core_args(parser)
    parser.add_argument("--history-limit", type=int, default=None)
    parser.add_argument("--relation-depth", type=int, default=None)
    parser.add_argument("--append-results", action="store_true", default=False)
    parser.add_argument("--append-results-csv", default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.set_defaults(func=command_relbench_run)


def add_relbench_inspect_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("inspect", help="Inspect materialized RelBench records and schema")
    parser.add_argument("--tasks", default="rel-f1:driver-dnf")
    parser.add_argument("--cache-dir", type=Path, default=None)
    add_bool_pair(parser, "download", dest="download")
    parser.add_argument("--history-limit", type=int, default=None)
    parser.add_argument("--relation-depth", type=int, default=None)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--fake", choices=("binary", "regression"), default=None)
    parser.set_defaults(func=command_relbench_inspect)


def add_relbench_report_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("report", help="Summarize a RelBench results CSV")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.set_defaults(func=command_relbench_report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified json2vec benchmark CLI")
    suite_parsers = parser.add_subparsers(dest="suite", required=True)

    tabarena_parser = suite_parsers.add_parser("tabarena", help="TabArena benchmark commands")
    tabarena_subparsers = tabarena_parser.add_subparsers(dest="command", required=True)
    add_tabarena_run_parser(tabarena_subparsers)

    relbench_parser = suite_parsers.add_parser("relbench", help="RelBench benchmark commands")
    relbench_subparsers = relbench_parser.add_subparsers(dest="command", required=True)
    add_relbench_run_parser(relbench_subparsers)
    add_relbench_inspect_parser(relbench_subparsers)
    add_relbench_report_parser(relbench_subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
