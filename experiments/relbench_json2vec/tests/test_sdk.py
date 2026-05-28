from __future__ import annotations

from pathlib import Path

from relbench_json2vec.config import JSON2VecConfig
from relbench_json2vec.sdk import BenchmarkReport, BenchmarkRun, BenchmarkSpec


def _model_config() -> JSON2VecConfig:
    return JSON2VecConfig(
        d_model=8,
        batch_size=2,
        max_epochs=1,
        lr=0.01,
        weight_decay=0.0,
        accelerator="cpu",
        attention="none",
        n_layers=1,
        n_linear=1,
        n_heads=4,
        random_seed=0,
        history_limit=3,
        relation_depth=1,
        max_cat_vocab_size=32,
    )


def test_benchmark_run_dry_run_writes_manifest(tmp_path: Path) -> None:
    spec = BenchmarkSpec(
        tasks=(("rel-f1", "driver-dnf"),),
        model_config=_model_config(),
        cache_dir=tmp_path / "cache",
        output_root=tmp_path / "runs",
        download=False,
    )
    run = BenchmarkRun(spec, run_id="fixed-run")
    manifest = run.dry_run()

    assert manifest["run_id"] == "fixed-run"
    assert manifest["status"] == "dry-run"
    assert run.manifest_path.exists()


def test_report_ranks_higher_and_lower_metrics(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    path.write_text(
        "\n".join(
            [
                "scope,dataset,task,task_type,model_or_config,split,metric,value,time_train_s,time_infer_s,notes",
                "s,d,t,binary_classification,majority,val,accuracy,0.4,1,1,",
                "s,d,t,binary_classification,json2vec_x,val,accuracy,0.6,1,1,",
                "s,d,t,regression,global_mean,test,mae,3.0,1,1,",
                "s,d,t,regression,json2vec_x,test,mae,2.0,1,1,",
            ]
        )
        + "\n"
    )
    report = BenchmarkReport.from_csv(path)
    verdicts = {(item.split, item.metric): item.verdict for item in report.comparisons}
    assert verdicts[("val", "accuracy")] == "better"
    assert verdicts[("test", "mae")] == "better"
