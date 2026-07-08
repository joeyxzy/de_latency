#! /home/joeyxzy/miniconda3/envs/vllm-0200/bin/python
"""
Plot experimental results from run_experiments.py.

Produces:
  1. metrics_vs_cores.png  — Benchtime, TPOT, Mean OS Delay vs n_cores
  2. gpu_util_{n}cores.png — Per-core GPU utilization over time (one per n_cores)

Usage: python3 plot_results.py [results_dir]
"""

import json
import csv
import sys
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_DIR = REPO_ROOT / "plots"


def load_results(results_dir: Path) -> dict:
    """Load all_results.json."""
    path = results_dir / "all_results.json"
    with open(path) as f:
        return json.load(f)


def load_gpu_csv(csv_path: str) -> dict:
    """Load GPU utilization CSV. Returns { col_name: [values] }. None if error."""
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return None

    if not rows:
        return None

    data = {}
    for col in rows[0].keys():
        data[col] = []
    for row in rows:
        for col, val in row.items():
            try:
                data[col].append(float(val))
            except ValueError:
                data[col].append(val)
    return data


# ── Figure 1: Metrics vs Cores ──────────────────────────────────────────────


def plot_metrics(results_dir: Path, results: dict) -> None:
    """Plot benchtime, TPOT, mean OS delay vs n_cores."""
    core_order = [1, 2, 3, 4, 8]
    n_cores_list = []
    benchtime_list = []
    tpot_list = []
    os_delay_list = []

    for nc in core_order:
        key = str(nc)
        if key not in results:
            continue
        r = results[key]
        if r.get("status") != "ok":
            continue
        n_cores_list.append(nc)
        benchtime_list.append(r["benchtime_s"])
        tpot_list.append(r["tpot_ms"])
        os_delay_list.append(r["mean_os_delay_ms"])

    if not n_cores_list:
        print("No valid results to plot.")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Benchtime
    ax1.plot(n_cores_list, benchtime_list, "o-", color="#2563EB", linewidth=2,
             markersize=8, markerfacecolor="white", markeredgewidth=2)
    ax1.set_ylabel("Benchtime (s)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    for x, y in zip(n_cores_list, benchtime_list):
        ax1.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)

    # TPOT
    ax2.plot(n_cores_list, tpot_list, "s-", color="#DC2626", linewidth=2,
             markersize=8, markerfacecolor="white", markeredgewidth=2)
    ax2.set_ylabel("Mean TPOT (ms)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    for x, y in zip(n_cores_list, tpot_list):
        ax2.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)

    # Mean OS Delay
    ax3.plot(n_cores_list, os_delay_list, "D-", color="#7C3AED", linewidth=2,
             markersize=8, markerfacecolor="white", markeredgewidth=2)
    ax3.set_xlabel("Number of CPU Cores", fontsize=12)
    ax3.set_ylabel("Mean OS Delay (ms)", fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(core_order)
    for x, y in zip(n_cores_list, os_delay_list):
        ax3.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)

    fig.suptitle("vLLM Performance vs CPU Core Count", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "metrics_vs_cores.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure 2-6: GPU Utilization per core config ─────────────────────────────


def plot_gpu_utilization(results_dir: Path, results: dict) -> None:
    """Plot GPU utilization over time: one grid figure per core config, one subplot per GPU."""
    core_order = [1, 2, 3, 4, 8]
    gpu_colors = [
        "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728",
        "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
    ]

    # ── Per-core: 4×2 grid (8 GPUs each in its own subplot) ─────────────
    for nc in core_order:
        key = str(nc)
        if key not in results:
            continue
        r = results[key]
        gpu_csv = r.get("gpu_csv", "")
        data = load_gpu_csv(gpu_csv)
        if data is None:
            print(f"No GPU data for {nc}cores, skipping.")
            continue

        timestamps = data.get("timestamp", [])
        if timestamps and isinstance(timestamps[0], str):
            x = list(range(len(timestamps)))
        else:
            x = list(range(len(data.get("gpu0_util", []))))

        fig, axes = plt.subplots(4, 2, figsize=(18, 12))
        axes = axes.flatten()

        for gpu_id in range(8):
            ax = axes[gpu_id]
            col = f"gpu{gpu_id}_util"
            values = data.get(col, [])
            if values:
                ax.plot(x[:len(values)], values,
                        color=gpu_colors[gpu_id],
                        linewidth=0.7)
                ax.fill_between(x[:len(values)], values, alpha=0.08,
                                color=gpu_colors[gpu_id])
            ax.set_title(f"GPU {gpu_id}", fontsize=10, fontweight="bold")
            ax.set_ylabel("Util (%)", fontsize=8)
            ax.set_ylim(-2, 105)
            ax.grid(True, alpha=0.2)

        for ax in axes[-2:]:
            ax.set_xlabel("Sample", fontsize=9)

        fig.suptitle(f"GPU Utilization — {nc} CPU Cores", fontsize=14, fontweight="bold")
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"gpu_util_{nc}cores.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    # ── Combined overview: one row per n_cores, each GPU as separate subplot ─
    n_valid = sum(1 for nc in core_order if str(nc) in results)
    if n_valid == 0:
        return

    fig, axes = plt.subplots(n_valid, 8, figsize=(32, 4 * n_valid))
    if n_valid == 1:
        axes = [axes]
    row = 0
    for nc in core_order:
        key = str(nc)
        if key not in results:
            continue
        r = results[key]
        gpu_csv = r.get("gpu_csv", "")
        data = load_gpu_csv(gpu_csv)
        if data is None:
            row += 1
            continue

        timestamps = data.get("timestamp", [])
        if timestamps and isinstance(timestamps[0], str):
            x = list(range(len(timestamps)))
        else:
            x = list(range(len(data.get("gpu0_util", []))))

        for gpu_id in range(8):
            ax = axes[row][gpu_id]
            col = f"gpu{gpu_id}_util"
            values = data.get(col, [])
            if values:
                ax.plot(x[:len(values)], values,
                        color=gpu_colors[gpu_id],
                        linewidth=0.35)
                ax.fill_between(x[:len(values)], values, alpha=0.04,
                                color=gpu_colors[gpu_id])
            ax.set_ylim(-2, 105)
            ax.grid(True, alpha=0.15)
            if row == 0:
                ax.set_title(f"GPU {gpu_id}", fontsize=8, fontweight="bold")
        axes[row][0].set_ylabel(f"{nc} cores\nUtil (%)", fontsize=7)

        row += 1

    for gpu_id in range(8):
        axes[-1][gpu_id].set_xlabel("Sample", fontsize=7)

    fig.suptitle("GPU Utilization — All Core Configurations", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "gpu_util_combined.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_dir}")
    results = load_results(results_dir)
    print(f"Found {len(results)} experiment results.")

    plot_metrics(results_dir, results)
    plot_gpu_utilization(results_dir, results)
    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
