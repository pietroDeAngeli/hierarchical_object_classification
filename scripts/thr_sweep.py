"""Threshold sweep for the hierarchical risk-coverage curve.

Trains the static hierarchy memory once, then evaluates it at multiple
acceptance-threshold values (--thr-a-values / --thr-a-linspace) and writes a
JSON array suitable for plotting R_H vs. coverage.

Usage example
-------------
PYTHONPATH=. python scripts/thr_sweep.py \
    --descriptor descriptor.json \
    --obj-mem-args input_static/obj_mem_args_random.json \
    --test-size 0.15 \
    --seed 0 \
    --thr-a-linspace 0.0 1.0 41 \
    --output ablation_results/thr_sweep_random.json
"""

from __future__ import annotations

import argparse
import json
import logging
import numpy as np
from pathlib import Path

from recsiam import init_hierarchy as ih


def _parse_thr_values(args) -> list[float]:
    if args.thr_a_values:
        return sorted(set(args.thr_a_values))
    lo, hi, n = args.thr_a_linspace
    return [float(v) for v in np.linspace(lo, hi, int(n))]


def main(args):
    obj_mem_args = ih._load_obj_mem_args(args.obj_mem_args,
                                         descriptor_path=args.descriptor)

    if args.evm_batch_size is not None:
        obj_mem_args["evm_batch_size"] = args.evm_batch_size

    samples = ih.build_samples_from_descriptor(args.descriptor)
    train_samples, test_samples = ih.split_fixed_samples(
        samples,
        test_size=args.test_size,
        train_size=args.train_size,
        seed=args.seed,
    )

    obj_mem = ih.memory_from_descriptor(
        args.descriptor,
        obj_mem_args,
        num_elements=len(train_samples),
        flat_hierarchy=args.flat_hierarchy,
    )

    jl_transform = None
    if args.jl_dim is not None:
        first_path = samples[0]["path"]
        probe = ih._load_embedding(first_path)
        in_dim = probe.shape[-1]
        jl_transform = ih._make_jl_projection(in_dim, args.jl_dim, seed=args.jl_seed)
        logging.info("JL projection: %d → %d dims (seed=%d)", in_dim, args.jl_dim, args.jl_seed)

    ih.fit_memory_from_samples(
        obj_mem,
        train_samples,
        log_every=args.train_log_every,
        jl_transform=jl_transform,
    )

    # Build ground-truth tree for geodesic / hierarchical metrics
    from recsiam import utils
    gt_tree = None
    if args.flat_hierarchy:
        try:
            _desc = utils.load_descriptor(args.descriptor)
            _hier = utils.hierarchy_from_descriptor(_desc)
            gt_tree = utils.tree_from_list(_hier)
        except Exception as exc:
            logging.warning("Could not load GT hierarchy: %s", exc)

    thr_values = _parse_thr_values(args)
    logging.info("Sweeping %d threshold values: %s", len(thr_values), thr_values)

    loader = (
        (lambda p: ih._load_embedding(p, jl_transform=jl_transform))
        if jl_transform is not None
        else ih._load_embedding
    )

    results = []
    for thr_a in thr_values:
        logging.info("Evaluating thr_a=%.4f ...", thr_a)
        metrics = ih.evaluate_test_samples(
            obj_mem,
            test_samples,
            thr_a=thr_a,
            thr_r=args.thr_r,
            jl_transform=jl_transform,
            gt_tree=gt_tree,
        )
        results.append({"thr_a": thr_a, "metrics": metrics})
        logging.info(
            "  thr_a=%.4f  acc=%.4f  h_acc=%.4f  cov=%.4f  hAURC=%s",
            thr_a,
            metrics.get("accuracy") or 0.0,
            metrics.get("hierarchical_accuracy") or 0.0,
            metrics.get("mean_coverage") or 0.0,
            f"{metrics['hAURC']:.4f}" if metrics.get("hAURC") is not None else "N/A",
        )

    payload = {
        "descriptor": str(args.descriptor),
        "obj_mem_args": str(args.obj_mem_args),
        "test_size": args.test_size,
        "seed": args.seed,
        "flat_hierarchy": args.flat_hierarchy,
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "sweep": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=1)
    logging.info("Saved sweep results to %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep thr_a for the hierarchical risk-coverage curve"
    )
    parser.add_argument("--descriptor", required=True)
    parser.add_argument("--obj-mem-args", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-size", default=0.15, type=float)
    parser.add_argument("--train-size", default=None, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--flat-hierarchy", action="store_true")
    parser.add_argument("--thr-r", default=None, type=float,
                        help="rejection threshold (default: same as thr_a)")
    parser.add_argument("--evm-batch-size", default=None, type=int)
    parser.add_argument("--train-log-every", default=100, type=int)
    parser.add_argument("--jl-dim", default=None, type=int)
    parser.add_argument("--jl-seed", default=0, type=int)

    thr_group = parser.add_mutually_exclusive_group(required=True)
    thr_group.add_argument(
        "--thr-a-values", nargs="+", type=float, metavar="V",
        help="explicit list of thr_a values to sweep, e.g. 0.0 0.1 0.5 0.9",
    )
    thr_group.add_argument(
        "--thr-a-linspace", nargs=3, type=float, metavar=("LO", "HI", "N"),
        help="sweep N evenly-spaced values from LO to HI, e.g. 0.0 1.0 41",
    )

    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

    main(args)
