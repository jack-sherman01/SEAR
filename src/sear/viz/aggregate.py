"""Aggregate results across multiple seeds and generate summary statistics + LaTeX table.

Usage (Python):
    from sear.viz.aggregate import aggregate_runs, latex_table
    stats = aggregate_runs(["runs/nailing_v3", "runs/nailing_default_s1", ...])
    print(latex_table(stats))

CLI:
    sear aggregate runs/nailing_v3 runs/nailing_default_s1 ... [--latex]
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

METHOD_ORDER = ["B0", "B1", "B1R", "B2", "B3", "SPE"]

# Metrics to aggregate: (summary_key_or_callable, display_name, format_spec)
# format_spec: "pct" = percentage, "int" = integer, "float2" = 2 decimal, "ratio" = n/m
_SCALAR_METRICS: list[tuple[str, str, str]] = [
    ("false_acceptance_rate",          "FAR(raw)",       "pct"),
    ("false_acceptance_rate_deployed", "FAR(dep)",       "pct"),
    ("deployment_violations",          "dep.viol",       "int"),
    ("rollbacks",                      "rollbacks",      "int"),
    ("admit_rate",                     "admit",          "pct"),
    ("reject_rate",                    "reject",         "pct"),
    ("mean_verification_s",            "verif(s)",       "float1"),
]
_SUITE_METRICS: list[tuple[str, str, str, str]] = [
    # (suite, key, display_name, format)
    ("in_distribution", "success_rate",      "ID succ",    "pct"),
    ("ood",             "success_rate",      "OOD succ",   "pct"),
    ("adversarial",     "violation_rate",    "ADV viol",   "pct"),
    ("adversarial",     "violation_upper",   "ADV ub",     "pct"),
    ("in_distribution", "mean_safety_margin","ID margin",  "float3"),
    ("in_distribution", "min_distance",      "min dist",   "float3"),
]
_RETENTION_METRICS: list[tuple[str, str]] = [
    ("in_distribution", "ID mret"),
    ("ood",             "OOD mret"),
]


def _safe(v: Any) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else math.nan
    except (TypeError, ValueError):
        return math.nan


def _load_summary(run_dir: Path, method: str) -> dict[str, Any] | None:
    f = run_dir / method / "result.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())["summary"]


def aggregate_runs(run_dirs: list[str | Path]) -> dict[str, dict[str, Any]]:
    """Aggregate summaries across seeds for each method.

    Returns: {method: {metric: {"mean": float, "std": float, "values": list}}}
    """
    run_dirs = [Path(d) for d in run_dirs]
    result: dict[str, dict[str, dict]] = {}

    for method in METHOD_ORDER:
        summaries = [s for d in run_dirs if (s := _load_summary(d, method)) is not None]
        if not summaries:
            continue

        metrics: dict[str, dict] = {}

        for key, name, _ in _SCALAR_METRICS:
            vals = [_safe(s.get(key, math.nan)) for s in summaries]
            vals = [v for v in vals if not math.isnan(v)]
            metrics[name] = {
                "mean": float(np.mean(vals)) if vals else math.nan,
                "std":  float(np.std(vals))  if len(vals) > 1 else 0.0,
                "values": vals,
                "n": len(vals),
            }

        for suite, key, name, _ in _SUITE_METRICS:
            vals = [_safe(s.get("final", {}).get(suite, {}).get(key, math.nan)) for s in summaries]
            vals = [v for v in vals if not math.isnan(v)]
            metrics[name] = {
                "mean": float(np.mean(vals)) if vals else math.nan,
                "std":  float(np.std(vals))  if len(vals) > 1 else 0.0,
                "values": vals,
                "n": len(vals),
            }

        for suite, name in _RETENTION_METRICS:
            vals = [_safe(s.get("margin_retention", {}).get(suite, math.nan)) for s in summaries]
            vals = [v for v in vals if not math.isnan(v)]
            metrics[name] = {
                "mean": float(np.mean(vals)) if vals else math.nan,
                "std":  float(np.std(vals))  if len(vals) > 1 else 0.0,
                "values": vals,
                "n": len(vals),
            }

        # Unsafe admitted counts (raw totals across seeds)
        metrics["_unsafe_admitted"] = sum(s.get("unsafe_admitted", 0) for s in summaries)
        metrics["_unsafe_proposals"] = sum(s.get("unsafe_proposals", 0) for s in summaries)
        metrics["_unsafe_admitted_dep"] = sum(s.get("unsafe_admitted_deployed", 0) for s in summaries)
        metrics["_unsafe_proposals_dep"] = sum(s.get("unsafe_proposals_deployed", 0) for s in summaries)
        metrics["_n_seeds"] = len(summaries)

        result[method] = metrics

    return result


def _fmt_cell(m: dict, fmt: str, show_std: bool = True) -> str:
    mean, std = m["mean"], m["std"]
    if math.isnan(mean):
        return "---"
    if fmt == "pct":
        s = f"{mean:.2f}"
        if show_std and std > 0:
            s += f"±{std:.2f}"
    elif fmt == "int":
        s = f"{mean:.1f}"
        if show_std and std > 0:
            s += f"±{std:.1f}"
    elif fmt == "float1":
        s = f"{mean:.1f}"
        if show_std and std > 0:
            s += f"±{std:.1f}"
    elif fmt == "float3":
        s = f"{mean:.3f}"
        if show_std and std > 0:
            s += f"±{std:.3f}"
    else:
        s = f"{mean:.2f}"
        if show_std and std > 0:
            s += f"±{std:.2f}"
    return s


def print_aggregate_report(stats: dict, n_seeds: int | None = None) -> None:
    """Print an ASCII aggregate report."""
    all_metrics = (
        [(name, fmt) for _, name, fmt in _SCALAR_METRICS]
        + [(name, fmt) for _, _, name, fmt in _SUITE_METRICS]
        + [(name, "float2") for _, name in _RETENTION_METRICS]
    )
    header = ["method", "unsafe adm.(raw)", "unsafe adm.(dep)"] + [n for n, _ in all_metrics]
    rows = []
    for method in METHOD_ORDER:
        if method not in stats:
            continue
        m = stats[method]
        ns = m.get("_n_seeds", 1)
        raw = f"{m['_unsafe_admitted']}/{m['_unsafe_proposals']}"
        dep = f"{m['_unsafe_admitted_dep']}/{m['_unsafe_proposals_dep']}"
        row = [f"{method}(n={ns})", raw, dep]
        for name, fmt in all_metrics:
            if name in m:
                row.append(_fmt_cell(m[name], fmt))
            else:
                row.append("---")
        rows.append(row)

    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    for r in [header] + rows:
        print("  ".join(str(v).rjust(w) for v, w in zip(r, widths)))


def latex_table(stats: dict, caption: str = "", label: str = "tab:results") -> str:
    """Generate a LaTeX table string from aggregated stats."""
    methods = [m for m in METHOD_ORDER if m in stats]

    # Column groups matching the proposal metrics
    col_defs = [
        ("unsafe adm.(raw)",  "_unsafe_admitted",     "_unsafe_proposals",     None,    "count"),
        ("unsafe adm.(dep)",  "_unsafe_admitted_dep", "_unsafe_proposals_dep", None,    "count"),
        ("FAR(raw)",          "FAR(raw)",              None,                   "pct",   "metric"),
        ("FAR(dep)",          "FAR(dep)",              None,                   "pct",   "metric"),
        ("dep.viol",          "dep.viol",              None,                   "int",   "metric"),
        ("rollbk",            "rollbacks",             None,                   "int",   "metric"),
        ("admit",             "admit",                 None,                   "pct",   "metric"),
        ("reject",            "reject",                None,                   "pct",   "metric"),
        ("ID succ",           "ID succ",               None,                   "pct",   "metric"),
        ("OOD succ",          "OOD succ",              None,                   "pct",   "metric"),
        ("ADV viol",          "ADV viol",              None,                   "pct",   "metric"),
        ("ADV ub",            "ADV ub",                None,                   "pct",   "metric"),
        ("ID mret",           "ID mret",               None,                   "float2","metric"),
        ("OOD mret",          "OOD mret",              None,                   "float2","metric"),
        ("verif(s)",          "verif(s)",              None,                   "float1","metric"),
    ]

    n_cols = 1 + len(col_defs)
    col_spec = "l" + "r" * len(col_defs)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{" + (caption or (
            rf"Multi-seed results (mean$\pm$std over {max((stats[m].get('_n_seeds', 1) for m in stats), default=1)} seeds)."
        )) + "}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Method & " + " & ".join(c[0] for c in col_defs) + r" \\",
        r"\midrule",
    ]

    for method in methods:
        m = stats[method]
        cells = [rf"\textbf{{{method}}}" if method == "SPE" else method]
        for col_name, key1, key2, fmt, kind in col_defs:
            if kind == "count":
                v1 = m.get(key1, 0)
                v2 = m.get(key2, 0)
                cells.append(f"{v1}/{v2}")
            else:
                if key1 in m:
                    cells.append(_fmt_cell(m[key1], fmt or "float2", show_std=True))
                else:
                    cells.append("---")
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)
