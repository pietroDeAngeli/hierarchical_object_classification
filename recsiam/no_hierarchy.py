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


def _group_by_target(samples: Sequence[dict]) -> Dict[object, List[dict]]:
    grouped: Dict[object, List[dict]] = {}
    for sample in samples:
        grouped.setdefault(sample["target"], []).append(sample)
    return grouped


def _concat_embeddings(samples: Sequence[dict]) -> np.ndarray:
    chunks = []
    for sample in samples:
        emb = init_h._load_embedding(sample["path"])
        emb = np.asarray(emb)
        if emb.ndim == 1:
            emb = emb[None, :]
        chunks.append(emb)

    if len(chunks) == 0:
        return np.array([])

    return np.concatenate(chunks, axis=0)


def _batched(seq: Sequence[dict], batch_size: int):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def train_classwise_evm(train_samples: Sequence[dict], evm_args: dict, batch_size: int = 64):
    grouped = _group_by_target(train_samples)
    classes = sorted(grouped.keys())

    if len(classes) < 2:
        raise ValueError("At least 2 classes are required for classwise one-vs-rest training")

    models = {}
    for cls_name in classes:
        pos_samples = grouped[cls_name]
        neg_samples = []
        for other_cls in classes:
            if other_cls == cls_name:
                continue
            neg_samples.extend(grouped[other_cls])

        neg_X = _concat_embeddings(neg_samples)
        if neg_X.size == 0:
            raise ValueError("No negative samples available for class '{}'".format(cls_name))

        cls_model = evm.EVM(**evm_args)
        seen = 0
        for batch in _batched(pos_samples, batch_size):
            pos_X = _concat_embeddings(batch)
            y = np.tile(cls_name, pos_X.shape[0]).astype(object)
            cls_model.fit(pos_X, y, neg_X=neg_X)
            seen += len(batch)

        models[cls_name] = cls_model

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


def evaluate_classwise_evm(models, test_samples):
    true_labels = []
    pred_labels = []

    for sample in test_samples:
        emb = init_h._load_embedding(sample["path"])
        pred = predict_with_classwise_evm(models, emb)
        true_labels.append(sample["target"])
        pred_labels.append(pred)

    return rec_eval.compute_eval_metrics(true_labels, pred_labels, tree=None)


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
    obj_mem_args = init_h._load_obj_mem_args(cmdline.obj_mem_args, descriptor_path=cmdline.descriptor)
    evm_args = obj_mem_args.get("evm_args")
    if not isinstance(evm_args, dict):
        raise ValueError("Could not find 'evm_args' in --obj-mem-args payload")

    samples = init_h.build_samples_from_descriptor(cmdline.descriptor)
    train_samples, test_samples = init_h.split_fixed_samples(
        samples,
        test_size=cmdline.test_size,
        train_size=cmdline.train_size,
        seed=cmdline.seed,
    )

    if len(train_samples) == 0:
        raise ValueError("No train samples available after split")

    models = train_classwise_evm(train_samples, evm_args=evm_args, batch_size=cmdline.batch_size)

    summary = {
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "classes": len(models),
        "batch_size": int(cmdline.batch_size),
        "seed": int(cmdline.seed),
    }

    if cmdline.eval_test:
        metrics = evaluate_classwise_evm(models, test_samples)
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
    parser.add_argument("--test-size", default=0, type=int, help="fixed number of samples reserved for test")
    parser.add_argument(
        "--train-size",
        default=None,
        type=int,
        help="fixed number of train samples after test split (default: all remaining)",
    )
    parser.add_argument("--seed", default=0, type=int, help="seed for deterministic train/test split")
    parser.add_argument("--batch-size", default=64, type=int, help="positive batch size used in classwise training")
    parser.add_argument("--eval-test", action="store_true", help="evaluate on test split and report metrics")
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
