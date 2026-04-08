import argparse
import csv
import json
import statistics
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

from pso import PSOOptimizer
from ssl_pipeline import run_pseudo_labeling_ssl


# These represent a small set of hand-picked settings a person might try manually.
MANUAL_CANDIDATE_CONFIGS: List[Dict[str, float]] = [
    {"threshold": 0.93, "max_pseudo_labels_per_round": 300, "pseudo_weight": 0.50},
    {"threshold": 0.95, "max_pseudo_labels_per_round": 500, "pseudo_weight": 0.75},
    {"threshold": 0.97, "max_pseudo_labels_per_round": 500, "pseudo_weight": 1.00},
    {"threshold": 0.95, "max_pseudo_labels_per_round": 1000, "pseudo_weight": 0.50},
    {"threshold": 0.98, "max_pseudo_labels_per_round": 300, "pseudo_weight": 1.00},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare manual hyperparameter tuning against PSO for the SSL pipeline."
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument(
        "--labeled-ratios",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3, 0.4, 0.5],
        help="Labeled-data fractions to evaluate."
    )
    parser.add_argument("--trials", type=int, default=5, help="Number of random seeds per ratio.")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--tuning-ssl-rounds", type=int, default=2)
    parser.add_argument("--tuning-epochs-per-round", type=int, default=3)
    parser.add_argument("--final-ssl-rounds", type=int, default=3)
    parser.add_argument("--final-epochs-per-round", type=int, default=4)
    parser.add_argument("--pso-swarm-size", type=int, default=3)
    parser.add_argument("--pso-iters", type=int, default=3)
    parser.add_argument("--pso-w", type=float, default=0.7)
    parser.add_argument("--pso-c1", type=float, default=1.5)
    parser.add_argument("--pso-c2", type=float, default=1.5)
    parser.add_argument(
        "--max-manual-candidates",
        type=int,
        default=None,
        help="Limit the number of manual candidate settings, mainly for faster smoke tests."
    )
    parser.add_argument(
        "--output-root",
        default="study_outputs",
        help="Parent directory where the report tables will be written."
    )
    return parser.parse_args()


def get_device_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if torch.backends.mps.is_available():
        return "Apple MPS"
    return "CPU"


def mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def stdev_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_markdown_table(rows: List[Dict], columns: List[str]) -> str:
    if not rows:
        return "No rows generated.\n"

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")

    return "\n".join(lines) + "\n"


def build_ssl_kwargs(args, labeled_ratio: float, seed: int, ssl_rounds: int, epochs_per_round: int) -> Dict:
    return {
        "data_dir": args.data_dir,
        "batch_size": args.batch_size,
        "val_ratio": args.val_ratio,
        "labeled_ratio": labeled_ratio,
        "seed": seed,
        "num_workers": args.num_workers,
        "ssl_rounds": ssl_rounds,
        "epochs_per_round": epochs_per_round,
        "learning_rate": args.learning_rate,
        "save_model": False,
        "verbose": False,
    }


def pick_best_manual_config(args, labeled_ratio: float, seed: int, manual_configs: List[Dict]) -> Dict:
    tuning_kwargs = build_ssl_kwargs(
        args=args,
        labeled_ratio=labeled_ratio,
        seed=seed,
        ssl_rounds=args.tuning_ssl_rounds,
        epochs_per_round=args.tuning_epochs_per_round
    )

    best_record = None

    for index, config in enumerate(manual_configs, start=1):
        result = run_pseudo_labeling_ssl(
            **tuning_kwargs,
            threshold=config["threshold"],
            max_pseudo_labels_per_round=config["max_pseudo_labels_per_round"],
            pseudo_weight=config["pseudo_weight"]
        )

        candidate_record = {
            "candidate_index": index,
            "config": dict(config),
            "best_val_acc": result["best_val_acc"],
            "test_acc": result["test_acc"],
            "test_f1": result["test_f1"],
            "total_pseudo_labeled": result["total_pseudo_labeled"],
        }

        if best_record is None or candidate_record["best_val_acc"] > best_record["best_val_acc"]:
            best_record = candidate_record

    return best_record


def run_manual_trial(args, labeled_ratio: float, seed: int, manual_configs: List[Dict]) -> Dict:
    tuning_start = time.perf_counter()
    best_manual = pick_best_manual_config(args, labeled_ratio, seed, manual_configs)
    tuning_time = time.perf_counter() - tuning_start

    final_kwargs = build_ssl_kwargs(
        args=args,
        labeled_ratio=labeled_ratio,
        seed=seed,
        ssl_rounds=args.final_ssl_rounds,
        epochs_per_round=args.final_epochs_per_round
    )

    final_start = time.perf_counter()
    final_result = run_pseudo_labeling_ssl(
        **final_kwargs,
        threshold=best_manual["config"]["threshold"],
        max_pseudo_labels_per_round=best_manual["config"]["max_pseudo_labels_per_round"],
        pseudo_weight=best_manual["config"]["pseudo_weight"]
    )
    final_time = time.perf_counter() - final_start

    return {
        "method": "manual",
        "selected_threshold": best_manual["config"]["threshold"],
        "selected_max_pseudo": best_manual["config"]["max_pseudo_labels_per_round"],
        "selected_pseudo_weight": best_manual["config"]["pseudo_weight"],
        "tuning_best_val_acc": best_manual["best_val_acc"],
        "final_best_val_acc": final_result["best_val_acc"],
        "final_test_acc": final_result["test_acc"],
        "final_test_f1": final_result["test_f1"],
        "total_pseudo_labeled": final_result["total_pseudo_labeled"],
        "tuning_time_sec": tuning_time,
        "final_run_time_sec": final_time,
        "total_time_sec": tuning_time + final_time,
        "search_history": "",
    }


def run_pso_trial(args, labeled_ratio: float, seed: int) -> Dict:
    tuning_ssl_kwargs = build_ssl_kwargs(
        args=args,
        labeled_ratio=labeled_ratio,
        seed=seed,
        ssl_rounds=args.tuning_ssl_rounds,
        epochs_per_round=args.tuning_epochs_per_round
    )

    tuning_start = time.perf_counter()
    optimizer = PSOOptimizer(
        swarm_size=args.pso_swarm_size,
        max_iters=args.pso_iters,
        w=args.pso_w,
        c1=args.pso_c1,
        c2=args.pso_c2,
        seed=seed,
        ssl_kwargs=tuning_ssl_kwargs,
        verbose=False
    )
    search_result = optimizer.optimize()
    tuning_time = time.perf_counter() - tuning_start

    final_kwargs = build_ssl_kwargs(
        args=args,
        labeled_ratio=labeled_ratio,
        seed=seed,
        ssl_rounds=args.final_ssl_rounds,
        epochs_per_round=args.final_epochs_per_round
    )

    final_start = time.perf_counter()
    final_result = run_pseudo_labeling_ssl(
        **final_kwargs,
        threshold=search_result["best_threshold"],
        max_pseudo_labels_per_round=search_result["best_max_pseudo"],
        pseudo_weight=search_result["best_pseudo_weight"]
    )
    final_time = time.perf_counter() - final_start

    return {
        "method": "pso",
        "selected_threshold": search_result["best_threshold"],
        "selected_max_pseudo": search_result["best_max_pseudo"],
        "selected_pseudo_weight": search_result["best_pseudo_weight"],
        "tuning_best_val_acc": search_result["best_fitness"],
        "final_best_val_acc": final_result["best_val_acc"],
        "final_test_acc": final_result["test_acc"],
        "final_test_f1": final_result["test_f1"],
        "total_pseudo_labeled": final_result["total_pseudo_labeled"],
        "tuning_time_sec": tuning_time,
        "final_run_time_sec": final_time,
        "total_time_sec": tuning_time + final_time,
        "search_history": json.dumps(search_result["history"]),
    }


def summarize_by_method(rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["labeled_ratio"], row["method"])].append(row)

    summary_rows = []
    for labeled_ratio, method in sorted(grouped):
        group = grouped[(labeled_ratio, method)]
        summary_rows.append({
            "labeled_ratio": labeled_ratio,
            "method": method,
            "trials": len(group),
            "mean_final_best_val_acc": round(mean_or_zero(r["final_best_val_acc"] for r in group), 4),
            "std_final_best_val_acc": round(stdev_or_zero(r["final_best_val_acc"] for r in group), 4),
            "mean_final_test_acc": round(mean_or_zero(r["final_test_acc"] for r in group), 4),
            "std_final_test_acc": round(stdev_or_zero(r["final_test_acc"] for r in group), 4),
            "mean_final_test_f1": round(mean_or_zero(r["final_test_f1"] for r in group), 4),
            "std_final_test_f1": round(stdev_or_zero(r["final_test_f1"] for r in group), 4),
            "mean_total_pseudo_labeled": round(mean_or_zero(r["total_pseudo_labeled"] for r in group), 1),
            "mean_tuning_time_sec": round(mean_or_zero(r["tuning_time_sec"] for r in group), 2),
            "mean_total_time_sec": round(mean_or_zero(r["total_time_sec"] for r in group), 2),
        })

    return summary_rows


def summarize_by_ratio(summary_rows: List[Dict]) -> List[Dict]:
    manual_map = {}
    pso_map = {}

    for row in summary_rows:
        key = row["labeled_ratio"]
        if row["method"] == "manual":
            manual_map[key] = row
        elif row["method"] == "pso":
            pso_map[key] = row

    comparison_rows = []
    for labeled_ratio in sorted(set(manual_map) | set(pso_map)):
        manual_row = manual_map.get(labeled_ratio, {})
        pso_row = pso_map.get(labeled_ratio, {})

        manual_acc = manual_row.get("mean_final_test_acc", 0.0)
        pso_acc = pso_row.get("mean_final_test_acc", 0.0)
        manual_f1 = manual_row.get("mean_final_test_f1", 0.0)
        pso_f1 = pso_row.get("mean_final_test_f1", 0.0)

        comparison_rows.append({
            "labeled_ratio": labeled_ratio,
            "manual_mean_test_acc": manual_acc,
            "pso_mean_test_acc": pso_acc,
            "pso_minus_manual_acc": round(pso_acc - manual_acc, 4),
            "manual_mean_test_f1": manual_f1,
            "pso_mean_test_f1": pso_f1,
            "pso_minus_manual_f1": round(pso_f1 - manual_f1, 4),
            "manual_mean_total_time_sec": manual_row.get("mean_total_time_sec", 0.0),
            "pso_mean_total_time_sec": pso_row.get("mean_total_time_sec", 0.0),
        })

    return comparison_rows


def save_report_tables(output_dir: Path, detailed_rows: List[Dict], summary_rows: List[Dict], comparison_rows: List[Dict]) -> None:
    detailed_columns = [
        "labeled_ratio",
        "trial",
        "seed",
        "method",
        "selected_threshold",
        "selected_max_pseudo",
        "selected_pseudo_weight",
        "tuning_best_val_acc",
        "final_best_val_acc",
        "final_test_acc",
        "final_test_f1",
        "total_pseudo_labeled",
        "tuning_time_sec",
        "final_run_time_sec",
        "total_time_sec",
        "search_history",
    ]
    write_csv(output_dir / "detailed_results.csv", detailed_rows, detailed_columns)

    summary_columns = [
        "labeled_ratio",
        "method",
        "trials",
        "mean_final_best_val_acc",
        "std_final_best_val_acc",
        "mean_final_test_acc",
        "std_final_test_acc",
        "mean_final_test_f1",
        "std_final_test_f1",
        "mean_total_pseudo_labeled",
        "mean_tuning_time_sec",
        "mean_total_time_sec",
    ]
    write_csv(output_dir / "summary_by_method.csv", summary_rows, summary_columns)

    comparison_columns = [
        "labeled_ratio",
        "manual_mean_test_acc",
        "pso_mean_test_acc",
        "pso_minus_manual_acc",
        "manual_mean_test_f1",
        "pso_mean_test_f1",
        "pso_minus_manual_f1",
        "manual_mean_total_time_sec",
        "pso_mean_total_time_sec",
    ]
    write_csv(output_dir / "summary_by_ratio.csv", comparison_rows, comparison_columns)

    markdown = [
        "# Manual vs PSO Report Tables",
        "",
        "## Summary By Method",
        "",
        make_markdown_table(summary_rows, summary_columns),
        "",
        "## Summary By Labeled Ratio",
        "",
        make_markdown_table(comparison_rows, comparison_columns),
    ]
    (output_dir / "report_tables.md").write_text("\n".join(markdown), encoding="utf-8")


def save_study_config(args, output_dir: Path, manual_configs: List[Dict]) -> None:
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": get_device_name(),
        "manual_candidate_configs": manual_configs,
        "args": vars(args),
    }
    (output_dir / "study_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    manual_configs = MANUAL_CANDIDATE_CONFIGS[:args.max_manual_candidates]
    if not manual_configs:
        raise ValueError("At least one manual candidate configuration is required.")

    visible_deprecation_warning = getattr(np, "VisibleDeprecationWarning", None)
    if visible_deprecation_warning is not None:
        warnings.filterwarnings("ignore", category=visible_deprecation_warning)
    warnings.filterwarnings(
        "ignore",
        message=r"dtype\(\): align should be passed as Python or NumPy boolean",
        category=DeprecationWarning
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"manual_vs_pso_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_study_config(args, output_dir, manual_configs)

    detailed_rows = []
    total_trials = len(args.labeled_ratios) * args.trials
    completed = 0

    for labeled_ratio in args.labeled_ratios:
        for trial_index in range(args.trials):
            seed = args.base_seed + trial_index
            completed += 1
            print(
                f"[{completed}/{total_trials}] labeled_ratio={labeled_ratio:.2f} "
                f"trial={trial_index + 1}/{args.trials} seed={seed}"
            )

            manual_result = run_manual_trial(args, labeled_ratio, seed, manual_configs)
            detailed_rows.append({
                "labeled_ratio": labeled_ratio,
                "trial": trial_index + 1,
                "seed": seed,
                **manual_result,
            })
            print(
                f"  manual -> test_acc={manual_result['final_test_acc']:.4f}, "
                f"test_f1={manual_result['final_test_f1']:.4f}, "
                f"total_time_sec={manual_result['total_time_sec']:.2f}"
            )

            pso_result = run_pso_trial(args, labeled_ratio, seed)
            detailed_rows.append({
                "labeled_ratio": labeled_ratio,
                "trial": trial_index + 1,
                "seed": seed,
                **pso_result,
            })
            print(
                f"  pso    -> test_acc={pso_result['final_test_acc']:.4f}, "
                f"test_f1={pso_result['final_test_f1']:.4f}, "
                f"total_time_sec={pso_result['total_time_sec']:.2f}"
            )

            summary_rows = summarize_by_method(detailed_rows)
            comparison_rows = summarize_by_ratio(summary_rows)
            save_report_tables(output_dir, detailed_rows, summary_rows, comparison_rows)

    summary_rows = summarize_by_method(detailed_rows)
    comparison_rows = summarize_by_ratio(summary_rows)
    save_report_tables(output_dir, detailed_rows, summary_rows, comparison_rows)

    print(f"\nSaved study outputs to: {output_dir.resolve()}")
    print(f"Detailed results: {(output_dir / 'detailed_results.csv').resolve()}")
    print(f"Summary tables  : {(output_dir / 'report_tables.md').resolve()}")


if __name__ == "__main__":
    main()
