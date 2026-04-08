import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a matplotlib report image from compare_manual_vs_pso study outputs."
    )
    parser.add_argument(
        "--study-dir",
        type=str,
        default=None,
        help="Path to a study output folder containing summary_by_ratio.csv."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="study_outputs",
        help="Parent directory used to auto-detect the newest study folder."
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="study_report_overview.png",
        help="Name of the generated overview matplotlib image."
    )
    return parser.parse_args()


def find_latest_study_dir(output_root: Path) -> Path:
    candidates = sorted(
        [path for path in output_root.glob("manual_vs_pso_*") if path.is_dir()]
    )
    if not candidates:
        raise FileNotFoundError(f"No study folders found in {output_root}")
    return candidates[-1]


def load_summary_rows(summary_path: Path) -> List[dict]:
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_per_epoch_rows(per_epoch_path: Path) -> List[dict]:
    with per_epoch_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(row: dict, key: str) -> float:
    return float(row[key])


def render_overview(rows: List[dict], destination: Path) -> None:
    labeled_percent = [int(round(to_float(row, "labeled_ratio") * 100)) for row in rows]
    manual_acc = [to_float(row, "manual_mean_test_acc") for row in rows]
    pso_acc = [to_float(row, "pso_mean_test_acc") for row in rows]
    acc_gain = [to_float(row, "pso_minus_manual_acc") for row in rows]

    table_headers = [
        "Labeled %",
        "Manual Acc",
        "PSO Acc",
        "PSO - Manual",
        "Manual Time (s)",
        "PSO Time (s)",
    ]
    table_rows = [
        [
            f"{percent}%",
            f"{to_float(row, 'manual_mean_test_acc'):.4f}",
            f"{to_float(row, 'pso_mean_test_acc'):.4f}",
            f"{to_float(row, 'pso_minus_manual_acc'):+.4f}",
            f"{to_float(row, 'manual_mean_total_time_sec'):.1f}",
            f"{to_float(row, 'pso_mean_total_time_sec'):.1f}",
        ]
        for percent, row in zip(labeled_percent, rows)
    ]

    fig = plt.figure(figsize=(13, 9))
    grid = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.3])

    ax_plot = fig.add_subplot(grid[0])
    ax_plot.plot(labeled_percent, manual_acc, marker="o", linewidth=2, label="Manual Tuning")
    ax_plot.plot(labeled_percent, pso_acc, marker="s", linewidth=2, label="PSO Tuning")
    ax_plot.bar(labeled_percent, acc_gain, alpha=0.18, width=4, label="PSO Gain")
    ax_plot.set_title("Manual vs PSO Hyperparameter Tuning Overview")
    ax_plot.set_xlabel("Labeled Data Percentage")
    ax_plot.set_ylabel("Mean Test Accuracy")
    ax_plot.set_xticks(labeled_percent)
    ax_plot.grid(True, linestyle="--", alpha=0.35)
    ax_plot.legend()

    ax_table = fig.add_subplot(grid[1])
    ax_table.axis("off")
    table = ax_table.table(
        cellText=table_rows,
        colLabels=table_headers,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    ax_table.set_title("Summary Table", pad=10)

    fig.tight_layout()
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)


def aggregate_convergence(per_epoch_rows: List[dict], labeled_ratio: float) -> List[Dict]:
    grouped = defaultdict(list)

    for row in per_epoch_rows:
        row_ratio = round(float(row["labeled_ratio"]), 4)
        if row_ratio != round(labeled_ratio, 4):
            continue
        grouped[(row["method"], int(row["global_epoch"]))].append(row)

    epoch_numbers = sorted({epoch for _, epoch in grouped.keys()})
    convergence_rows = []

    for epoch in epoch_numbers:
        manual_rows = grouped.get(("manual", epoch), [])
        pso_rows = grouped.get(("pso", epoch), [])

        convergence_rows.append({
            "global_epoch": epoch,
            "manual_mean_val_acc": (
                sum(float(row["val_acc"]) for row in manual_rows) / len(manual_rows)
                if manual_rows else 0.0
            ),
            "pso_mean_val_acc": (
                sum(float(row["val_acc"]) for row in pso_rows) / len(pso_rows)
                if pso_rows else 0.0
            ),
            "manual_mean_improvement": (
                sum(float(row["val_acc_improvement"]) for row in manual_rows) / len(manual_rows)
                if manual_rows else 0.0
            ),
            "pso_mean_improvement": (
                sum(float(row["val_acc_improvement"]) for row in pso_rows) / len(pso_rows)
                if pso_rows else 0.0
            ),
        })

    return convergence_rows


def render_convergence_report(convergence_rows: List[Dict], labeled_ratio: float, destination: Path) -> None:
    epochs = [row["global_epoch"] for row in convergence_rows]
    manual_acc = [row["manual_mean_val_acc"] for row in convergence_rows]
    pso_acc = [row["pso_mean_val_acc"] for row in convergence_rows]

    table_headers = [
        "Epoch",
        "Manual Val Acc",
        "PSO Val Acc",
        "Manual Δ",
        "PSO Δ",
    ]
    table_rows = [
        [
            row["global_epoch"],
            f"{row['manual_mean_val_acc']:.4f}",
            f"{row['pso_mean_val_acc']:.4f}",
            f"{row['manual_mean_improvement']:+.4f}",
            f"{row['pso_mean_improvement']:+.4f}",
        ]
        for row in convergence_rows
    ]

    fig = plt.figure(figsize=(13, 9))
    grid = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.3])

    ax_plot = fig.add_subplot(grid[0])
    ax_plot.plot(epochs, manual_acc, marker="o", linewidth=2, label="Manual")
    ax_plot.plot(epochs, pso_acc, marker="s", linewidth=2, label="PSO")
    ax_plot.set_title(f"Convergence at {int(round(labeled_ratio * 100))}% Labeled Data")
    ax_plot.set_xlabel("Global Epoch")
    ax_plot.set_ylabel("Mean Validation Accuracy")
    ax_plot.set_xticks(epochs)
    ax_plot.grid(True, linestyle="--", alpha=0.35)
    ax_plot.legend()

    ax_table = fig.add_subplot(grid[1])
    ax_table.axis("off")
    table = ax_table.table(
        cellText=table_rows,
        colLabels=table_headers,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)
    ax_table.set_title("Per-Epoch Improvement Table", pad=10)

    fig.tight_layout()
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_reports_for_study(study_dir: Path, output_name: str = "study_report_overview.png") -> List[Path]:
    summary_path = study_dir / "summary_by_ratio.csv"
    per_epoch_path = study_dir / "per_epoch_results.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find {summary_path}")
    if not per_epoch_path.exists():
        raise FileNotFoundError(f"Could not find {per_epoch_path}")

    summary_rows = load_summary_rows(summary_path)
    per_epoch_rows = load_per_epoch_rows(per_epoch_path)
    if not summary_rows:
        raise ValueError(f"No rows found in {summary_path}")
    if not per_epoch_rows:
        raise ValueError(f"No rows found in {per_epoch_path}")

    created_paths = []
    destination = study_dir / output_name
    render_overview(summary_rows, destination)
    created_paths.append(destination.resolve())

    labeled_ratios = sorted({float(row["labeled_ratio"]) for row in per_epoch_rows})
    for labeled_ratio in labeled_ratios:
        convergence_rows = aggregate_convergence(per_epoch_rows, labeled_ratio)
        if not convergence_rows:
            continue

        ratio_destination = study_dir / f"study_report_ratio_{int(round(labeled_ratio * 100))}.png"
        render_convergence_report(convergence_rows, labeled_ratio, ratio_destination)
        created_paths.append(ratio_destination.resolve())

    return created_paths


def main():
    args = parse_args()
    if args.study_dir is not None:
        study_dir = Path(args.study_dir)
    else:
        study_dir = find_latest_study_dir(Path(args.output_root))

    created_paths = render_reports_for_study(study_dir, args.output_name)
    for path in created_paths:
        print(path)


if __name__ == "__main__":
    main()
