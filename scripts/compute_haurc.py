"""Compute hAURC from a thr_sweep.py output JSON.

The sweep gives one (mean_coverage, hierarchical_accuracy) point per thr_a value.
Varying thr_a traces a risk-coverage curve:

    R_H(lambda) = 1 - Acc_H(lambda)
    hAURC = integral R_H(c) dc   (trapz over sweep points, sorted by coverage)

Usage
-----
python scripts/compute_haurc.py ablation_results/thr_sweep_random.json
python scripts/compute_haurc.py ablation_results/thr_sweep_*.json --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_sweep(path: str):
    with open(path) as f:
        data = json.load(f)
    covs, risks, thrs, h_accs = [], [], [], []
    for entry in data["sweep"]:
        m = entry["metrics"]
        cov = m.get("mean_coverage")
        hacc = m.get("hierarchical_accuracy")
        if cov is None or hacc is None:
            continue
        covs.append(cov)
        risks.append(1.0 - hacc)
        thrs.append(entry["thr_a"])
        h_accs.append(hacc)
    return covs, risks, thrs, h_accs, data


def trapz(xs, ys):
    """Trapezoidal integration of y over x (lists, not numpy)."""
    xs, ys = zip(*sorted(zip(xs, ys)))
    total = 0.0
    for i in range(len(xs) - 1):
        total += 0.5 * (ys[i] + ys[i + 1]) * abs(xs[i + 1] - xs[i])
    return total


def compute_haurc(covs, risks):
    """Area under R_H(c) curve via trapezoidal rule."""
    if len(covs) < 2:
        return None
    return trapz(covs, risks)


def print_table(path, covs, risks, thrs, h_accs, haurc):
    print(f"\n=== {Path(path).name} ===")
    print(f"  hAURC (sweep) = {haurc:.6f}")
    print()
    print(f"  {'thr_a':>8}  {'coverage':>10}  {'h_acc':>10}  {'h_risk':>10}")
    print("  " + "-" * 44)
    for thr, cov, hacc, risk in sorted(zip(thrs, covs, h_accs, risks)):
        print(f"  {thr:8.4f}  {cov:10.4f}  {hacc:10.4f}  {risk:10.4f}")


def main(args):
    results = []
    for path in args.inputs:
        covs, risks, thrs, h_accs, data = load_sweep(path)
        haurc = compute_haurc(covs, risks)
        results.append((path, covs, risks, thrs, h_accs, haurc, data))
        print_table(path, covs, risks, thrs, h_accs, haurc)

    if len(results) > 1:
        print("\n=== Summary ===")
        for path, _, _, _, _, haurc, _ in results:
            label = Path(path).stem
            print(f"  {label:<40}  hAURC = {haurc:.6f}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for --plot: pip install matplotlib", file=sys.stderr)
            sys.exit(1)

        fig, ax = plt.subplots(figsize=(6, 4))
        labels = args.label if args.label else [Path(p).stem for p in args.inputs]
        labels += [Path(p).stem for p in args.inputs[len(labels):]]

        for (path, covs, risks, thrs, _, haurc, _), label in zip(results, labels):
            pts = sorted(zip(covs, risks))
            cs, rs = zip(*pts)
            ax.plot(cs, rs, marker="o", markersize=4,
                    label=f"{label} (hAURC={haurc:.4f})")

        ax.set_xlabel("Mean coverage $c$")
        ax.set_ylabel("Hierarchical risk $R_H = 1 - \\mathrm{Acc}_H$")
        ax.set_title("Hierarchical risk-coverage curve")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        out = Path(args.plot_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150)
        print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", metavar="JSON",
                        help="one or more thr_sweep output JSON files")
    parser.add_argument("--plot", action="store_true",
                        help="also generate a risk-coverage plot")
    parser.add_argument("--label", nargs="+", default=[],
                        help="legend labels (one per input file)")
    parser.add_argument("--plot-output", default="ablation_results/risk_coverage.png",
                        metavar="PATH")
    main(parser.parse_args())
