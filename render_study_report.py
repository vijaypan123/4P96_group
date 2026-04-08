import argparse
import csv
from pathlib import Path
from typing import List

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
        default="study_report.png",
        help="Name of the generated matplotlib image."
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


def to_float(row: dict, key: str) -> float:
    return float(row[key])


def render_report(rows: List[dict], destination: Path) -> None:
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
    ax_plot.set_title("Manual vs PSO Hyperparameter Tuning")
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


def main():
    args = parse_args()
    if args.study_dir is not None:
        study_dir = Path(args.study_dir)
    else:
        study_dir = find_latest_study_dir(Path(args.output_root))

    summary_path = study_dir / "summary_by_ratio.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find {summary_path}")

    rows = load_summary_rows(summary_path)
    if not rows:
        raise ValueError(f"No rows found in {summary_path}")

    destination = study_dir / args.output_name
    render_report(rows, destination)
    print(destination.resolve())


if __name__ == "__main__":
    main()
