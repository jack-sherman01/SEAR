"""Command-line interface.

    sear run CONFIG [--methods B0 SPE] [--out runs/x] [--seed N]  run an evolution study
    sear report RUN_DIR                                compare methods of a finished run
    sear plot RUN_DIR                                  generate figures from a finished run
    sear render RUN_DIR [--n-episodes N] [--fps F]     render trajectory videos
    sear aggregate RUN_DIR... [--latex] [--out F]      aggregate multi-seed results
    sear nsim-plot --budgets 10 30 60 120 --dirs-10 runs/... --dirs-30 ... --out fig.pdf
    sear sign-contract CONTRACT.yaml [--key-file K]    sign a safety contract (safety authority)
    sear verify-ledger LEDGER.jsonl                    check an evidence ledger's hash chain
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from sear.contracts.signing import load_key, write_signed
from sear.evolution.experiment import METHODS, load_config, run_experiment
from sear.lineage.ledger import EvidenceLedger, LedgerIntegrityError

def _cycle(s: dict, suite: str = "in_distribution") -> str:
    t0 = s["initial"][suite].get("mean_completion_time")
    t1 = s["final"][suite].get("mean_completion_time")
    if t0 is None or t1 is None:
        return "-"
    try:
        return f"{float(t0):.2f}→{float(t1):.2f}s"
    except (TypeError, ValueError):
        return "-"


REPORT_COLUMNS = [
    ("unsafe adm.(raw)", lambda s: f"{s['unsafe_admitted']}/{s['unsafe_proposals']}"),
    ("unsafe adm.(dep)", lambda s: f"{s.get('unsafe_admitted_deployed', '?')}/{s.get('unsafe_proposals_deployed', '?')}"),
    ("FAR(raw)", lambda s: _fmt(s["false_acceptance_rate"])),
    ("FAR(dep)", lambda s: _fmt(s.get("false_acceptance_rate_deployed", math.nan))),
    ("admit", lambda s: _fmt(s["admit_rate"])),
    ("reject", lambda s: _fmt(s["reject_rate"])),
    ("rollbk", lambda s: str(s["rollbacks"])),
    ("dep.viol", lambda s: str(s["deployment_violations"])),
    ("ID succ", lambda s: _fmt(s["final"]["in_distribution"]["success_rate"])),
    ("OOD succ", lambda s: _fmt(s["final"]["ood"]["success_rate"])),
    ("ADV viol", lambda s: _fmt(s["final"]["adversarial"]["violation_rate"])),
    ("ADV ub", lambda s: _fmt(s["final"]["adversarial"]["violation_upper"])),
    ("min dist", lambda s: _fmt(s["final"]["in_distribution"].get("min_distance", math.nan), 3)),
    ("drop rate", lambda s: _fmt(s["final"]["adversarial"].get("tool_drop_rate", math.nan))),
    ("margin ret.", lambda s: _fmt(s["margin_retention"].get("in_distribution", math.nan))),
    ("cycle time", _cycle),
    ("verif s", lambda s: _fmt(s["mean_verification_s"], 1)),
]


def _fmt(x, digits: int = 2) -> str:
    try:
        return "-" if x is None or not math.isfinite(float(x)) else f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def print_report(summaries: list[dict]) -> None:
    header = ["method"] + [c for c, _ in REPORT_COLUMNS]
    rows = [[s["method"]] + [f(s) for _, f in REPORT_COLUMNS] for s in summaries]
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    for r in [header] + rows:
        print("  ".join(str(v).rjust(w) for v, w in zip(r, widths)))


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg = dict(cfg)
        cfg["seed"] = args.seed
    default_name = cfg.get("name", Path(args.config).stem)
    if args.seed is not None:
        default_name = f"{default_name}_s{args.seed}"
    out = args.out or Path("runs") / default_name
    results = run_experiment(cfg, args.methods, out)
    print()
    print_report([r.summary() for r in results.values()])
    print(f"\nresults written to {out}")
    # auto-generate figures
    try:
        from sear.viz.plots import generate_figures
        figs = generate_figures(out)
        print(f"figures:  {', '.join(str(f) for f in figs)}")
    except Exception as exc:
        print(f"[plot] skipped: {exc}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    files = sorted(Path(args.run_dir).glob("*/result.json"))
    if not files:
        print(f"no */result.json under {args.run_dir}", file=sys.stderr)
        return 1
    print_report([json.loads(f.read_text())["summary"] for f in files])
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    from sear.viz.plots import generate_figures
    try:
        figs = generate_figures(args.run_dir)
        for f in figs:
            print(f)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def cmd_render(args: argparse.Namespace) -> int:
    from sear.viz.render import render_episodes
    try:
        videos = render_episodes(args.run_dir, n_episodes=args.n_episodes,
                                 methods=args.methods, fps=args.fps)
        for v in videos:
            print(v)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def cmd_aggregate(args: argparse.Namespace) -> int:
    from sear.viz.aggregate import aggregate_runs, latex_table, print_aggregate_report
    from sear.viz.plots import generate_multiseed_figures
    stats = aggregate_runs(args.run_dirs)
    if not stats:
        print("no result.json files found", file=sys.stderr)
        return 1
    print_aggregate_report(stats)
    # shaded generation-curve figure
    out_dir = Path(args.run_dirs[0]) / "figures_multiseed"
    try:
        figs = generate_multiseed_figures(args.run_dirs, out_dir)
        print(f"\nfigures: {', '.join(str(f) for f in figs)}")
    except Exception as exc:
        print(f"[plot] skipped: {exc}")
    if args.latex:
        tex = latex_table(stats, caption=args.caption, label=args.label)
        if args.out:
            Path(args.out).write_text(tex)
            print(f"LaTeX table written to {args.out}")
        else:
            print("\n" + tex)
    return 0


def cmd_nsim_plot(args: argparse.Namespace) -> int:
    from sear.viz.plots import plot_nsim_sweep
    # dirs are provided in order matching budgets; split evenly or accept all dirs per budget
    # Simple interface: --budgets 10 30 60 120 --dirs d0 d1 d2 ... (all dirs, assigned to budgets by order)
    n = len(args.budgets)
    all_dirs = args.dirs
    chunk = len(all_dirs) // n if n else len(all_dirs)
    seed_dirs = {}
    for i, b in enumerate(args.budgets):
        seed_dirs[b] = all_dirs[i * chunk:(i + 1) * chunk]
    out = plot_nsim_sweep(args.budgets, seed_dirs, args.out)
    print(f"figure: {out}")
    return 0


def cmd_sign(args: argparse.Namespace) -> int:
    sig = write_signed(args.contract, load_key(key_file=args.key_file), key_id=args.key_id)
    print(f"signed -> {sig}")
    return 0


def cmd_verify_ledger(args: argparse.Namespace) -> int:
    if not Path(args.ledger).is_file():
        print(f"INVALID: ledger file {args.ledger} not found")
        return 1
    try:
        led = EvidenceLedger(args.ledger)
    except LedgerIntegrityError as exc:
        print(f"INVALID: {exc}")
        return 1
    kinds: dict[str, int] = {}
    for r in led.records():
        kinds[r.kind.value] = kinds.get(r.kind.value, 0) + 1
    print(f"OK: {len(led)} records, chain intact; {kinds}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sear", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run an evolution study")
    p.add_argument("config")
    p.add_argument("--methods", nargs="+", choices=sorted(METHODS), metavar="METHOD")
    p.add_argument("--out", type=Path)
    p.add_argument("--seed", type=int, default=None, help="override config seed")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="summarize a finished run")
    p.add_argument("run_dir")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("plot", help="generate figures from a finished run")
    p.add_argument("run_dir")
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("render", help="render trajectory videos from a finished run")
    p.add_argument("run_dir")
    p.add_argument("--n-episodes", type=int, default=3)
    p.add_argument("--methods", nargs="+")
    p.add_argument("--fps", type=int, default=20)
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("aggregate", help="aggregate multi-seed results and optionally emit LaTeX")
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--latex", action="store_true", help="also print/write a LaTeX table")
    p.add_argument("--out", default=None, help="write LaTeX to this file instead of stdout")
    p.add_argument("--caption", default="", help="LaTeX table caption")
    p.add_argument("--label", default="tab:results", help="LaTeX table label")
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser("nsim-plot", help="plot n_sim sweep: admit rate and FAR vs budget")
    p.add_argument("--budgets", nargs="+", type=int, required=True, metavar="N")
    p.add_argument("--out", default="runs/figures/nsim_sweep.pdf")
    for b_help in [("dirs", "space-separated run dirs for all budgets in order")]:
        p.add_argument(f"--{b_help[0]}", nargs="+", required=True, metavar="DIR", help=b_help[1])
    p.set_defaults(func=cmd_nsim_plot)

    p = sub.add_parser("sign-contract", help="sign a safety contract")
    p.add_argument("contract")
    p.add_argument("--key-file")
    p.add_argument("--key-id", default="default")
    p.set_defaults(func=cmd_sign)

    p = sub.add_parser("verify-ledger", help="verify an evidence ledger")
    p.add_argument("ledger")
    p.set_defaults(func=cmd_verify_ledger)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
