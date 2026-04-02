from __future__ import annotations

import logging

import networkx as nx
import numpy as np


def majority_vote(labels: np.ndarray):
    if labels.size == 0:
        return None
    uniq, cnt = np.unique(labels.astype(object), return_counts=True)
    return uniq[np.argmax(cnt)]


def compute_eval_metrics(true_labels, pred_labels, tree=None):
    logger = logging.getLogger("recsiam.eval.compute_eval_metrics")

    if len(true_labels) == 0:
        return {
            "accuracy": None,
            "precision": None,
            "f1": None,
            "mean_geodesic_distance": None,
        }

    true_labels = np.asarray(true_labels, dtype=object)
    pred_labels = np.asarray(pred_labels, dtype=object)

    accuracy = float(np.mean(true_labels == pred_labels))
    labels = np.unique(np.concatenate((true_labels, pred_labels)).astype(object))

    precisions = []
    f1_scores = []
    for lab in labels:
        tp = np.sum((pred_labels == lab) & (true_labels == lab))
        fp = np.sum((pred_labels == lab) & (true_labels != lab))
        fn = np.sum((pred_labels != lab) & (true_labels == lab))

        precision_i = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall_i = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        if (precision_i + recall_i) > 0.0:
            f1_i = 2.0 * precision_i * recall_i / (precision_i + recall_i)
        else:
            f1_i = 0.0

        precisions.append(precision_i)
        f1_scores.append(f1_i)

    precision = float(np.mean(precisions)) if len(precisions) > 0 else None
    f1 = float(np.mean(f1_scores)) if len(f1_scores) > 0 else None

    mean_geo = None
    if tree is not None and len(tree.nodes) > 0:
        undir = tree.to_undirected()
        geo_distances = []
        missing_node_pairs = 0
        no_path_pairs = 0
        for t_lab, p_lab in zip(true_labels, pred_labels):
            if t_lab not in undir.nodes or p_lab not in undir.nodes:
                missing_node_pairs += 1
                continue
            try:
                geo_distances.append(float(nx.shortest_path_length(undir, t_lab, p_lab)))
            except nx.NetworkXNoPath:
                no_path_pairs += 1
                continue
            except Exception:
                no_path_pairs += 1
                continue

        total_pairs = len(true_labels)
        used_pairs = len(geo_distances)
        logger.info(
            "Geodesic diagnostics: total_pairs=%d used_pairs=%d missing_node_pairs=%d no_path_pairs=%d",
            total_pairs,
            used_pairs,
            missing_node_pairs,
            no_path_pairs,
        )

        if len(geo_distances) > 0:
            mean_geo = float(np.mean(geo_distances))
        else:
            logger.warning(
                "mean_geodesic_distance is None because no valid (true,pred) pair mapped to the evaluation tree"
            )
    elif tree is None:
        logger.info("Geodesic diagnostics: tree is None, geodesic distance disabled for this evaluation")
    else:
        logger.warning("Geodesic diagnostics: evaluation tree is empty, geodesic distance disabled")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "f1": f1,
        "mean_geodesic_distance": mean_geo,
    }


def evaluate_test_samples(obj_mem, test_samples, embedding_loader, thr_a=0.0, thr_r=None, logger=None):
    if logger is None:
        logger = logging.getLogger("recsiam.eval")

    true_labels = []
    pred_labels = []

    for sample in test_samples:
        target = sample["target"]
        emb = embedding_loader(sample["path"])

        try:
            pred, _, _ = obj_mem.predict(emb, thr_a=thr_a, thr_r=thr_r)
        except Exception as exc:
            logger.warning("Skipping test sample '%s' due to prediction error: %s", sample["path"], exc)
            continue

        pred = np.asarray(pred, dtype=object)
        if pred.ndim == 0:
            pred = pred[None]

        sample_pred = majority_vote(pred)
        if sample_pred is None:
            continue

        true_labels.append(target)
        pred_labels.append(sample_pred)

    return compute_eval_metrics(true_labels, pred_labels, obj_mem.T)
