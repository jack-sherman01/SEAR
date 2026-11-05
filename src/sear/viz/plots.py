"""Generate figures from experiment result JSON files.

Outputs written to ``<run_dir>/figures/``:
  generation_curves.pdf  — per-suite metrics over generations, one line per method
  summary_bars.pdf       — FAR, violations, margin-retention bar chart
  task_score.pdf         — task score trajectory per method
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def _safe(v: Any) -> float:
    """Convert any value (including JSON string 'nan'/'inf') to float; return nan on failure."""
    try:
        f = float(v)
        return f if math.isfinite(f) else math.nan
    except (TypeError, ValueError):
        return math.nan


METHOD_COLORS = {
    "B0": "#888888",
    "B1": "#2196F3",
    "B2": "#FF9800",
    "B3": "#9C27B0",
    "SPE": "#4CAF50",
}
METHOD_ORDER = ["B0", "B1", "B2", "B3", "SPE"]


def _load_results(run_dir: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for f in sorted(run_dir.glob("*/result.json")):
        data = json.loads(f.read_text())
        results[data["method"]] = data
    return results


def _gen_series(data: dict, suite: str, metric: str) -> list[float]:
    return [_safe(r["suites"].get(suite, {}).get(metric, math.nan)) for r in data["records"]]


def _task_score_series(data: dict) -> list[float]:
    return [_safe(r.get("task_score", math.nan)) for r in data["records"]]


def _plot_line(ax, results: dict, suite: str, metric: str, ylabel: str) -> None:
    for method in METHOD_ORDER:
        if method not in results:
            continue
        ys = _gen_series(results[method], suite, metric)
        xs = list(range(len(ys)))
        ax.plot(xs, ys, label=method, color=METHOD_COLORS.get(method), marker="o",
                markersize=3, linewidth=1.5)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel("Generation", fontsize=8)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)


def plot_generation_curves(results: dict, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Evolution Metrics by Generation", fontsize=13, y=1.01)

    specs = [
        ("in_distribution", "success_rate",      "ID success rate"),
        ("held_out",         "success_rate",      "Held-out success rate"),
        ("ood",              "success_rate",      "OOD success rate"),
        ("adversarial",      "violation_rate",    "Adversarial violation rate"),
        ("in_distribution",  "mean_safety_margin","Safety margin (ID)"),
        ("adversarial",      "violation_upper",   "Adv. viol. upper (95%)"),
    ]
    for ax, (suite, metric, label) in zip(axes.flat, specs):
        _plot_line(ax, results, suite, metric, label)

    plt.tight_layout()
    out = out_dir / "generation_curves.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


def plot_task_score(results: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in METHOD_ORDER:
        if method not in results:
            continue
        ys = _task_score_series(results[method])
        ax.plot(range(len(ys)), ys, label=method, color=METHOD_COLORS.get(method),
                marker="o", markersize=3, linewidth=1.5)
    ax.set_title("Task Score over Generations", fontsize=11)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Task score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = out_dir / "task_score.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


def plot_summary_bars(results: dict, out_dir: Path) -> Path:
    methods = [m for m in METHOD_ORDER if m in results]
    summaries = {m: results[m]["summary"] for m in methods}
    colors = [METHOD_COLORS.get(m, "#aaa") for m in methods]
    xs = np.arange(len(methods))

    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    fig.suptitle("Summary Comparison", fontsize=12)

    def bar(ax, key, title, pct=False):
        vals = [_safe(summaries[m].get(key, math.nan)) for m in methods]
        heights = [0.0 if math.isnan(v) else v for v in vals]
        bars = ax.bar(xs, heights, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels(methods, rotation=30, fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        if pct:
            ax.set_ylim(0, 1.05)
        cap = max(heights) if any(h > 0 for h in heights) else 1.0
        for b, v in zip(bars, vals):
            if not math.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01 * cap,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    bar(axes[0], "false_acceptance_rate",          "FAR (raw)",         pct=True)
    bar(axes[1], "false_acceptance_rate_deployed",  "FAR (deployed)",   pct=True)
    bar(axes[2], "deployment_violations",           "Deployment violations")
    bar(axes[3], "rollbacks",                       "Rollbacks")

    # Margin retention grouped bar
    ax = axes[4]
    suites = ["in_distribution", "held_out", "ood", "adversarial"]
    labels = ["ID", "HO", "OOD", "ADV"]
    width = 0.15
    for i, m in enumerate(methods):
        mr = summaries[m].get("margin_retention", {})
        ys = [_safe(mr.get(s, math.nan)) for s in suites]
        heights = [0.0 if math.isnan(v) else v for v in ys]
        ax.bar(np.arange(len(suites)) + i * width, heights,
               width=width, label=m, color=METHOD_COLORS.get(m), alpha=0.85)
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.4, linewidth=1)
    ax.set_title("Margin retention", fontsize=9)
    ax.set_xticks(np.arange(len(suites)) + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = out_dir / "summary_bars.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


def generate_figures(run_dir: str | Path) -> list[Path]:
    """Generate all figures for a completed run directory; return list of output paths."""
    run_dir = Path(run_dir)
    results = _load_results(run_dir)
    if not results:
        raise ValueError(f"no */result.json found under {run_dir}")
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = [
        plot_generation_curves(results, out_dir),
        plot_task_score(results, out_dir),
        plot_summary_bars(results, out_dir),
    ]
    return generated
