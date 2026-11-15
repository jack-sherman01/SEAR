"""Analyze a hardware-path evolution run.

Usage:
    python scripts/analyze_hardware_run.py runs/hw_nailing_analysis

Produces:
    <run_dir>/figures/generation_curves.pdf
    <run_dir>/figures/summary_bars.pdf
    <run_dir>/figures/task_score.pdf
    <run_dir>/figures/hw_latency.pdf
    <run_dir>/figures/hw_calibration.pdf
    <run_dir>/hw_analysis.txt
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHOD_COLORS = {"B0": "#888", "B1": "#2196F3", "B2": "#FF9800", "B3": "#9C27B0", "SPE": "#4CAF50"}
METHOD_ORDER = ["B0", "B1", "B2", "B3", "SPE"]


def load(run_dir: Path) -> dict:
    out = {}
    for f in sorted(run_dir.glob("*/result.json")):
        d = json.loads(f.read_text())
        out[d["method"]] = d
    return out


def _safe(v) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else math.nan
    except (TypeError, ValueError):
        return math.nan


# ------------------------------------------------------------------ hw latency
def plot_hw_latency(results: dict, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Hardware Latency per Generation (deployment trials)", fontsize=11)

    for method in METHOD_ORDER:
        if method not in results:
            continue
        records = results[method]["records"]
        gens, means, maxes = [], [], []
        for rec in records:
            hw = rec.get("hw_latency_stats")
            if hw:
                gens.append(rec["generation"])
                means.append(_safe(hw.get("mean_ms")))
                maxes.append(_safe(hw.get("max_ms")))
        if not gens:
            continue
        c = METHOD_COLORS.get(method)
        axes[0].plot(gens, means, label=method, color=c, marker="o", markersize=3, linewidth=1.5)
        axes[1].plot(gens, maxes, label=method, color=c, marker="s", markersize=3, linewidth=1.5)

    for ax, title in zip(axes, ["Mean latency (ms)", "Peak latency (ms)"]):
        ax.set_xlabel("Generation", fontsize=9)
        ax.set_ylabel(title, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.axhline(20.0, color="red", linestyle="--", linewidth=0.9, alpha=0.7, label="fallback threshold")

    plt.tight_layout()
    out = out_dir / "hw_latency.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


# -------------------------------------------------- hw calibration verdicts
def plot_hw_calibration(results: dict, out_dir: Path) -> Path:
    verdict_map = {"pass": 2, "unknown": 1, "fail": 0}
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle("Tier 3.5 HardwareCalibrationCheck Verdicts per Generation", fontsize=11)

    for i, method in enumerate([m for m in METHOD_ORDER if m in results]):
        records = results[method]["records"]
        gens, scores = [], []
        for rec in records:
            v = rec.get("hw_calibration_verdict")
            if v:
                gens.append(rec["generation"])
                scores.append(verdict_map.get(v, 1))
        if not gens:
            continue
        c = METHOD_COLORS.get(method)
        ax.scatter(gens, [i + s * 0.1 for s in scores], color=c, label=method, s=60, zorder=3)
        ax.step(gens, [i + s * 0.1 for s in scores], color=c, alpha=0.4, linewidth=1)

    methods_present = [m for m in METHOD_ORDER if m in results
                       and any(r.get("hw_calibration_verdict") for r in results[m]["records"])]
    if not methods_present:
        ax.text(0.5, 0.5, "No Tier 3.5 verdicts recorded\n(B1/B2 skip verification; check SPE)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
    else:
        ax.set_yticks(range(len(methods_present)))
        ax.set_yticklabels(methods_present, fontsize=9)
        ax.set_xlabel("Generation", fontsize=9)
        ax.legend(fontsize=8)

    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out = out_dir / "hw_calibration.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ text report
def text_report(results: dict, run_dir: Path) -> Path:
    lines = [f"Hardware evolution analysis: {run_dir.name}", "=" * 60, ""]

    for method in METHOD_ORDER:
        if method not in results:
            continue
        s = results[method]["summary"]
        lines.append(f"Method: {method}")
        lines.append(f"  Generations:            {s.get('generations', '?')}")
        lines.append(f"  Admit rate:             {_safe(s.get('admit_rate')):.2f}")
        lines.append(f"  FAR (raw):              {_safe(s.get('false_acceptance_rate')):.3f}")
        lines.append(f"  FAR (deployed):         {_safe(s.get('false_acceptance_rate_deployed')):.3f}")
        lines.append(f"  Deployment violations:  {s.get('deployment_violations', 0)}")
        lines.append(f"  Rollbacks:              {s.get('rollbacks', 0)}")
        lines.append(f"  Mean verification (s):  {_safe(s.get('mean_verification_s')):.1f}")
        final = s.get("final", {}).get("in_distribution", {})
        lines.append(f"  Final ID success rate:  {_safe(final.get('success_rate')):.2f}")
        lines.append(f"  Final ID safety margin: {_safe(final.get('mean_safety_margin')):.4f}")
        margin_ret = s.get("margin_retention", {})
        lines.append(f"  Margin retention (ID):  {_safe(margin_ret.get('in_distribution')):.3f}")
        hw = s.get("hw_latency")
        if hw:
            lines.append(f"  HW mean latency (ms):   {_safe(hw.get('mean_ms')):.2f}")
            lines.append(f"  HW max latency (ms):    {_safe(hw.get('max_ms')):.2f}")
            lines.append(f"  HW latency fallbacks:   {hw.get('total_fallbacks', 0)}")
        verdicts = s.get("hw_calibration_verdicts", [])
        if verdicts:
            from collections import Counter
            vc = Counter(verdicts)
            lines.append(f"  Tier 3.5 verdicts:      {dict(vc)}")
        lines.append("")

    # comparison table
    lines.append("Comparison table")
    lines.append("-" * 50)
    header = f"{'Method':<8} {'FAR(raw)':>9} {'FAR(dep)':>9} {'Dep.Viol':>9} {'ID success':>11} {'HW lat(ms)':>11}"
    lines.append(header)
    for method in METHOD_ORDER:
        if method not in results:
            continue
        s = results[method]["summary"]
        hw = s.get("hw_latency", {}) or {}
        far_r = _safe(s.get("false_acceptance_rate"))
        far_d = _safe(s.get("false_acceptance_rate_deployed"))
        dv = s.get("deployment_violations", 0)
        ids = _safe(s.get("final", {}).get("in_distribution", {}).get("success_rate"))
        lat = _safe(hw.get("mean_ms"))
        far_r_s = f"{far_r:.3f}" if not math.isnan(far_r) else "  n/a"
        far_d_s = f"{far_d:.3f}" if not math.isnan(far_d) else "  n/a"
        lat_s = f"{lat:.2f}" if not math.isnan(lat) else "  n/a"
        lines.append(f"{method:<8} {far_r_s:>9} {far_d_s:>9} {dv:>9} {ids:>11.2f} {lat_s:>11}")

    out = run_dir / "hw_analysis.txt"
    out.write_text("\n".join(lines) + "\n")
    return out


# ------------------------------------------------------------------ main
def main(run_dir_str: str) -> None:
    run_dir = Path(run_dir_str)
    results = load(run_dir)
    if not results:
        print(f"No result.json files found under {run_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    from sear.viz.plots import generate_figures
    std_figs = generate_figures(run_dir)
    for f in std_figs:
        print(f"  {f}")

    hw_lat = plot_hw_latency(results, out_dir)
    hw_cal = plot_hw_calibration(results, out_dir)
    report = text_report(results, run_dir)

    print(f"  {hw_lat}")
    print(f"  {hw_cal}")
    print(f"  {report}")
    print(f"\nDone. Figures in {out_dir}/")

    # print summary to stdout
    txt = report.read_text()
    print("\n" + txt)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <run_dir>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
