"""Phase 5: compare two models on the clean and all degraded test sets.

Usage (Colab, CPU is fine):
    python -m sarship.compare --data-root DATA --runs-root RUNS
    [--set compare.baseline=baseline_yolo11n compare.candidate=robust_yolo11n]

Needs `python -m sarship.robustness` to have run for both models (it caches the
predictions this script reads). Every difference is a paired bootstrap CI on the
same images; the verdict follows the rule fixed in the config before training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sarship.analysis.ap import map_from_tables, match_all
from sarship.analysis.bootstrap import paired_bootstrap
from sarship.analysis.run import build_images, load_split
from sarship.config import DEFAULT_CONFIG, load_config
from sarship.robustness import condition_id, conditions

SEEN, HELD_LEVEL, HELD_TYPE, ALL_DEGRADED = (
    "seen (levels 1-4 of trained degradations)",
    "held-out severity (level 5)",
    "held-out degradation type",
    "all degraded conditions",
)


def condition_groups(cfg: dict) -> dict[str, list[str]]:
    """Which conditions the robust model saw in training and which test generalisation."""
    trained = set(cfg["robust_train"]["degradations"])
    groups: dict[str, list[str]] = {SEEN: [], HELD_LEVEL: [], HELD_TYPE: [], ALL_DEGRADED: []}
    for name, spec in cfg["robustness"]["degradations"].items():
        ids = [condition_id(name, float(v)) for v in spec["values"]]
        groups[ALL_DEGRADED] += ids
        if name in trained:
            groups[SEEN] += ids[:-1]
            groups[HELD_LEVEL].append(ids[-1])
        else:
            groups[HELD_TYPE] += ids
    return {g: ids for g, ids in groups.items() if ids}


def load_tables(items: list[dict], run_dir: Path, cid: str):
    cache = run_dir / "robustness" / "predictions" / f"{cid}.json"
    if not cache.exists():
        raise FileNotFoundError(f"{cache} missing: run `python -m sarship.robustness` "
                                f"for {run_dir.name} first")
    preds = json.loads(cache.read_text())["predictions"]
    return match_all(build_images(items, preds))


def decide(result: dict, cfg: dict) -> dict:
    c = cfg["compare"]
    clean = result["clean"]["map50_95"]
    degraded = result[ALL_DEGRADED]["map50_95"]
    clean_ok = clean["diff"] > -c["clean_tolerance"]
    gain_ok = degraded["low"] > 0
    return {"clean_ok": clean_ok, "degraded_gain_ok": gain_ok, "accept": clean_ok and gain_ok}


def compare(cfg: dict, items: list[dict], runs_root: Path) -> dict:
    c = cfg["compare"]
    run_a, run_b = runs_root / c["baseline"], runs_root / c["candidate"]
    ids = [condition_id(n, v) for n, v in conditions(cfg)]
    pairs, points = {}, {}
    for cid in ids:
        ta, tb = load_tables(items, run_a, cid), load_tables(items, run_b, cid)
        pairs[cid] = (ta, tb)
        points[cid] = {"baseline": map_from_tables(ta), "candidate": map_from_tables(tb)}
        print(f"{cid:>16}: baseline {points[cid]['baseline']['map50_95']:.3f}  "
              f"candidate {points[cid]['candidate']['map50_95']:.3f}", flush=True)

    groups = condition_groups(cfg)
    result = paired_bootstrap(pairs, groups, c["bootstrap_resamples"],
                              cfg["analysis"]["bootstrap_ci"], cfg["seed"])
    report = {"baseline": c["baseline"], "candidate": c["candidate"], "points": points,
              "differences": result, "groups": groups, "decision": decide(result, cfg)}
    report["markdown"] = report_markdown(report, cfg)
    return report


def _ci(d: dict) -> str:
    flag = " *" if d["low"] > 0 or d["high"] < 0 else ""
    return f"{d['diff']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]{flag}"


def report_markdown(report: dict, cfg: dict) -> str:
    pts, diff = report["points"], report["differences"]
    lines = [f"Baseline: `{report['baseline']}`, candidate: `{report['candidate']}`. "
             "Δ = candidate − baseline, paired 95% CI; * = CI excludes 0.", "",
             "| condition | baseline mAP@0.5:0.95 | candidate | Δ mAP@0.5:0.95 | Δ mAP@0.5 |",
             "|---|---|---|---|---|"]
    for cid in pts:
        lines.append(f"| {cid} | {pts[cid]['baseline']['map50_95']:.3f} | "
                     f"{pts[cid]['candidate']['map50_95']:.3f} | {_ci(diff[cid]['map50_95'])} | "
                     f"{_ci(diff[cid]['map50'])} |")
    lines += ["", "| group | conditions | Δ mAP@0.5:0.95 | Δ mAP@0.5 |", "|---|---|---|---|"]
    for g, members in report["groups"].items():
        lines.append(f"| {g} | {len(members)} | {_ci(diff[g]['map50_95'])} | "
                     f"{_ci(diff[g]['map50'])} |")
    d = report["decision"]
    verdict = "BETTER" if d["accept"] else "NOT shown to be better"
    tol = cfg["compare"]["clean_tolerance"]
    lines += ["", "Pre-registered decision:",
              f"- clean loss smaller than {tol}: {'yes' if d['clean_ok'] else 'NO'} "
              f"(Δ clean = {diff['clean']['map50_95']['diff']:+.3f})",
              f"- mean gain over all degraded conditions, CI above 0: "
              f"{'yes' if d['degraded_gain_ok'] else 'NO'}",
              f"- **verdict: candidate is {verdict}**"]
    return "\n".join(lines) + "\n"


def plot_comparison(report: dict, cfg: dict, out_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts, diff = report["points"], report["differences"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (name, spec) in zip(axes.flat, cfg["robustness"]["degradations"].items(),
                                strict=False):
        ids = ["clean"] + [condition_id(name, float(v)) for v in spec["values"]]
        x = np.arange(len(ids))
        for who, style in (("baseline", "o--"), ("candidate", "s-")):
            ax.plot(x, [pts[i][who]["map50_95"] for i in ids], style,
                    label=report[who] if who == "baseline" else report["candidate"])
        trained = name in cfg["robust_train"]["degradations"]
        ax.set_title(f"{name} ({'trained up to level 4' if trained else 'held out'})")
        ax.set_xticks(x, ["clean"] + [f"{v:g}" for v in spec["values"]])
        ax.set_xlabel(spec["param"])
        ax.set_ylabel("mAP@0.5:0.95")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")

    ax = axes.flat[5]
    keys = ["clean", *report["groups"]]
    vals = [diff[k]["map50_95"] for k in keys]
    y = np.arange(len(keys))
    ax.errorbar([v["diff"] for v in vals], y,
                xerr=[[v["diff"] - v["low"] for v in vals], [v["high"] - v["diff"] for v in vals]],
                fmt="o", capsize=4)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y, [k if len(k) < 30 else k[:27] + "..." for k in keys], fontsize=8)
    ax.set_xlabel("Δ mAP@0.5:0.95 (candidate − baseline), paired 95% CI")
    ax.set_title("Differences")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    items = load_split(args.data_root / cfg["dataset"]["yolo_dirname"], "test")
    report = compare(cfg, items, args.runs_root)
    out_dir = args.runs_root / cfg["compare"]["candidate"] / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.md").write_text(report["markdown"], encoding="utf-8")
    (out_dir / "comparison.json").write_text(
        json.dumps({k: v for k, v in report.items() if k != "markdown"}, indent=2))
    plot_comparison(report, cfg, out_dir / "comparison.png")
    print(report["markdown"])
    print(f"Figures and tables in {out_dir}")


if __name__ == "__main__":
    main()
