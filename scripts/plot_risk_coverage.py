"""Plot the hierarchical risk-coverage curve from a thr_sweep.py output JSON.

Usage
-----
python scripts/plot_risk_coverage.py \
    --input ablation_results/thr_sweep_random.json \
    --label "Random negatives" \
    --output ablation_results/risk_coverage.pdf

Multiple sweeps can be overlaid:
    python scripts/plot_risk_coverage.py \
        --input ablation_results/thr_sweep_random.json \
        --input ablation_results/thr_sweep_faiss.json \
        --label "Random" "FAISS" \
        --output ablation_results/risk_coverage.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_curve(path: str):
    """Return lists (coverages, risks, thr_values) from a thr_sweep JSON."""
    with open(path) as f:
        data = json.load(f)
    coverages, risks, thrs = [], [], []
    for entry in data["sweep"]:
        m = entry["metrics"]
        cov = m.get("mean_coverage")
        hacc = m.get("hierarchical_accuracy")
        if cov is None or hacc is None:
            continue
        coverages.append(cov)
        risks.append(1.0 - hacc)
        thrs.append(entry["thr_a"])
    return coverages, risks, thrs


def main(args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    labels = args.label if args.label else [Path(p).stem for p in args.input]
    if len(labels) < len(args.input):
        labels += [Path(p).stem for p in args.input[len(labels):]]

    multi = len(args.input) > 1

    fig, (ax_rc, ax_cov) = plt.subplots(1, 2, figsize=(11, 4))

    for path, label in zip(args.input, labels):
        covs, risks, thrs = load_curve(path)

        # --- left: risk-coverage curve (sorted by coverage ascending) ---
        pts = sorted(zip(covs, risks, thrs))
        covs_s, risks_s, _ = zip(*pts) if pts else ([], [], [])
        kw = dict(marker="o", markersize=3)
        if multi:
            kw["label"] = label
        ax_rc.plot(covs_s, risks_s, **kw)

        # --- right: coverage vs thr_a ---
        pts2 = sorted(zip(thrs, covs))
        thrs_s, covs_s2 = zip(*pts2) if pts2 else ([], [])
        kw2 = dict(marker="o", markersize=3)
        if multi:
            kw2["label"] = label
        ax_cov.plot(thrs_s, covs_s2, **kw2)

    ax_rc.set_xlabel("Mean coverage $c$")
    ax_rc.set_ylabel("Hierarchical risk $R_H$")
    ax_rc.set_title("Hierarchical risk-coverage curve")
    ax_rc.grid(True, linestyle="--", alpha=0.4)
    if multi:
        ax_rc.legend(fontsize=8)

    ax_cov.set_xlabel(r"Threshold $\lambda$")
    ax_cov.set_ylabel("Mean coverage $c$")
    ax_cov.set_title("Coverage vs. threshold")
    ax_cov.grid(True, linestyle="--", alpha=0.4)
    if multi:
        ax_cov.legend(fontsize=8)

    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    print(f"Saved plot to {out}")

    # Also print a compact table
    print(f"\n{'thr_a':>8}  {'coverage':>10}  {'h_risk':>10}")
    print("-" * 34)
    for path in args.input:
        covs, risks, thrs = load_curve(path)
        for thr, cov, risk in sorted(zip(thrs, covs, risks)):
            print(f"{thr:8.3f}  {cov:10.4f}  {risk:10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, metavar="JSON",
                        help="one or more thr_sweep output JSON files")
    parser.add_argument("--label", nargs="+", default=None, metavar="STR",
                        help="legend labels (one per --input file)")
    parser.add_argument("--output", default="ablation_results/risk_coverage.pdf")
    main(parser.parse_args())
