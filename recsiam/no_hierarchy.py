from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import lz4.frame
import numpy as np

from . import evm
from . import eval as rec_eval
from . import init_hierarchy as init_h
from . import utils


def _group_by_target(samples: Sequence[dict]) -> Dict[object, List[dict]]:
    grouped: Dict[object, List[dict]] = {}
    for sample in samples:
        grouped.setdefault(sample["target"], []).append(sample)
    return grouped


def _concat_embeddings(samples: Sequence[dict], jl_transform=None) -> np.ndarray:
    chunks = []
    for sample in samples:
        emb = init_h._load_embedding(sample["path"], jl_transform=jl_transform)
        emb = np.asarray(emb)
        if emb.ndim == 1:
            emb = emb[None, :]
        chunks.append(emb)

    if len(chunks) == 0:
        return np.array([])

    return np.concatenate(chunks, axis=0)


def train_classwise_evm(train_samples: Sequence[dict], evm_args: dict, batch_size: int = 64,
                        jl_transform=None):
    grouped = _group_by_target(train_samples)
    classes = sorted(grouped.keys())
    logging.info("Training classwise EVM: %d classes, %d total samples, batch_size=%d",
                 len(classes), len(train_samples), batch_size)

    if len(classes) < 2:
        raise ValueError("At least 2 classes are required for classwise one-vs-rest training")

    # Pre-load all embeddings once — avoids reloading from disk O(classes) times
    logging.info("Pre-loading all %d train embeddings...", len(train_samples))
    all_samples_flat: List[dict] = []
    class_bounds: Dict[object, tuple] = {}  # cls -> (start_row, end_row)
    offset = 0
    for cls_name in classes:
        n = len(grouped[cls_name])
        class_bounds[cls_name] = (offset, offset + n)
        all_samples_flat.extend(grouped[cls_name])
        offset += n
    all_X = _concat_embeddings(all_samples_flat, jl_transform=jl_transform)
    logging.info("Pre-loaded embeddings: shape=%s", all_X.shape)

    models = {}
    total_seen = 0
    for cls_idx, cls_name in enumerate(classes, 1):
        s, e = class_bounds[cls_name]
        pos_X_full = all_X[s:e]  # view, no copy

        # neg_X: all rows except this class's slice
        neg_parts = []
        if s > 0:
            neg_parts.append(all_X[:s])
        if e < len(all_X):
            neg_parts.append(all_X[e:])
        neg_X = np.concatenate(neg_parts, axis=0) if len(neg_parts) > 1 else neg_parts[0]

        n_pos = len(pos_X_full)
        logging.info("[%d/%d] class '%s': %d pos, %d neg samples",
                     cls_idx, len(classes), cls_name, n_pos, len(neg_X))

        cls_model = evm.EVM(**evm_args)
        seen = 0
        n_batches = (n_pos + batch_size - 1) // batch_size
        for batch_idx in range(1, n_batches + 1):
            bs = (batch_idx - 1) * batch_size
            be = min(bs + batch_size, n_pos)
            pos_X = pos_X_full[bs:be]  # view, no copy
            y = np.tile(cls_name, len(pos_X)).astype(object)
            cls_model.fit(pos_X, y, neg_X=neg_X)
            seen += len(pos_X)
            total_seen += len(pos_X)
            overall_pct = 100.0 * total_seen / len(train_samples)
            cls_pct = 100.0 * seen / n_pos
            logging.info(
                "  batch %d/%d — class '%s': %d/%d pos (%.1f%%) | overall %d/%d (%.1f%%)",
                batch_idx, n_batches, cls_name,
                seen, n_pos, cls_pct,
                total_seen, len(train_samples), overall_pct,
            )

        models[cls_name] = cls_model

    logging.info("Training complete: %d class models built", len(models))
    return models


def predict_with_classwise_evm(models: Dict[object, evm.EVM], emb: np.ndarray):
    emb = np.asarray(emb)
    if emb.ndim == 1:
        emb = emb[None, :]

    labels = list(models.keys())
    score_mat = np.empty((emb.shape[0], len(labels)), dtype=float)

    for j, label in enumerate(labels):
        _, _, probs = models[label].predict(emb, ret_distribution=True)
        probs = np.asarray(probs)
        if probs.ndim == 1:
            probs = probs[None, :]
        score_mat[:, j] = probs[:, 0]

    # max over samples (rows): for each class take its best per-frame score,
    # then pick the class with the highest peak confidence.
    class_scores = score_mat.max(axis=0)
    return labels[int(np.argmax(class_scores))]


def evaluate_classwise_evm(models, test_samples, jl_transform=None, gt_tree=None):
    true_labels = []
    pred_labels = []
    logging.info("Evaluating on %d test samples...", len(test_samples))

    for i, sample in enumerate(test_samples, 1):
        emb = init_h._load_embedding(sample["path"], jl_transform=jl_transform)
        pred = predict_with_classwise_evm(models, emb)
        true_labels.append(sample["target"])
        pred_labels.append(pred)
        if i % 100 == 0 or i == len(test_samples):
            logging.info("  evaluated %d/%d samples", i, len(test_samples))

    return rec_eval.compute_eval_metrics(true_labels, pred_labels, tree=None, gt_tree=gt_tree)


def _save_model(models, path_like):
    out_path = Path(path_like)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "classes": list(models.keys()),
        "models": {k: v.get_params() for k, v in models.items()},
    }
    with lz4.frame.open(str(out_path), mode="wb", compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC) as f:
        pickle.dump(payload, f)


def main(cmdline):
    logging.info("Loading obj-mem-args from %s", cmdline.obj_mem_args)
    obj_mem_args = init_h._load_obj_mem_args(cmdline.obj_mem_args, descriptor_path=cmdline.descriptor)
    evm_args = obj_mem_args.get("evm_args")
    if not isinstance(evm_args, dict):
        raise ValueError("Could not find 'evm_args' in --obj-mem-args payload")

    samples = init_h.build_samples_from_descriptor(cmdline.descriptor)
    logging.info("Loaded %d total samples from descriptor", len(samples))
    train_samples, test_samples = init_h.split_fixed_samples(
        samples,
        test_size=cmdline.test_size,
        train_size=cmdline.train_size,
        seed=cmdline.seed,
    )
    logging.info("Split: %d train, %d test (seed=%d)", len(train_samples), len(test_samples), cmdline.seed)

    if len(train_samples) == 0:
        raise ValueError("No train samples available after split")

    # Build JL projection matrix if requested
    jl_transform = None
    if cmdline.jl_dim is not None:
        probe = init_h._load_embedding(samples[0]["path"])
        in_dim = probe.shape[-1]
        jl_transform = init_h._make_jl_projection(in_dim, cmdline.jl_dim, seed=cmdline.jl_seed)
        logging.info("JL projection: %d → %d dims (seed=%d)", in_dim, cmdline.jl_dim, cmdline.jl_seed)

    models = train_classwise_evm(train_samples, evm_args=evm_args, batch_size=cmdline.batch_size,
                                 jl_transform=jl_transform)

    summary = {
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "classes": len(models),
        "batch_size": int(cmdline.batch_size),
        "seed": int(cmdline.seed),
    }

    gt_tree = None
    try:
        _desc = utils.load_descriptor(cmdline.descriptor)
        _hier = utils.hierarchy_from_descriptor(_desc)
        gt_tree = utils.tree_from_list(_hier)
        logging.info("Loaded GT hierarchy for geodesic distance (%d nodes)", len(gt_tree.nodes))
    except Exception as exc:
        logging.warning("Could not load GT hierarchy for geodesic distance: %s", exc)

    if cmdline.eval_test:
        metrics = evaluate_classwise_evm(models, test_samples, jl_transform=jl_transform, gt_tree=gt_tree)
        summary["test_metrics"] = metrics
        logging.info("Test metrics: %s", metrics)

    if cmdline.output is not None:
        out_path = Path(cmdline.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as ofile:
            json.dump(summary, ofile, indent=1)

    if cmdline.model_output is not None:
        _save_model(models, cmdline.model_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor", required=True, type=str, help="descriptor JSON path")
    parser.add_argument(
        "--obj-mem-args",
        required=True,
        type=str,
        help="JSON file with object memory args (must include evm_args)",
    )
    parser.add_argument("--output", default=None, type=str, help="optional output json summary path")
    parser.add_argument("--model-output", default=None, type=str, help="optional serialized model output (.lz4)")
    parser.add_argument("--test-size", default=0, type=float, help="number of test samples (>=1) or fraction of total (0<v<1, e.g. 0.15 for 15%%)")
    parser.add_argument(
        "--train-size",
        default=None,
        type=int,
        help="fixed number of train samples after test split (default: all remaining)",
    )
    parser.add_argument("--seed", default=0, type=int, help="seed for deterministic train/test split")
    parser.add_argument("--batch-size", default=64, type=int, help="positive batch size used in classwise training")
    parser.add_argument("--eval-test", action="store_true", help="evaluate on test split and report metrics")
    parser.add_argument("--jl-dim", default=None, type=int,
                        help="if set, apply a Gaussian Johnson–Lindenstrauss projection to reduce "
                             "each embedding from its original dimension to this many dimensions")
    parser.add_argument("--jl-seed", default=0, type=int,
                        help="random seed for the JL projection matrix (default: 0)")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    parser.add_argument("-q", "--quite", action="store_true", help="disable warnings")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quite:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

    main(args)
