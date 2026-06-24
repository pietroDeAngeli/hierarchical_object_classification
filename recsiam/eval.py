from __future__ import annotations

import logging

import networkx as nx
import numpy as np


def majority_vote(labels: np.ndarray):
    if labels.size == 0:
        return None
    uniq, cnt = np.unique(labels.astype(object), return_counts=True)
    return uniq[np.argmax(cnt)]


def _hierarchy_stats(tree):
    if tree is None or len(tree.nodes) == 0:
        return {
            "height": None,
            "avg_depth": None,
            "branching_factor_min": None,
            "branching_factor_max": None,
            "branching_factor_avg": None,
        }

    if hasattr(tree, "is_directed") and tree.is_directed():
        roots = [n for n in tree.nodes if tree.in_degree(n) == 0]
        if len(roots) == 0:
            roots = [next(iter(tree.nodes))]

        depths = {}
        for root in roots:
            for node, dist in nx.single_source_shortest_path_length(tree, root).items():
                if node not in depths or dist < depths[node]:
                    depths[node] = int(dist)

        if len(depths) < len(tree.nodes):
            # Fallback for malformed/disconnected directed structures.
            undir = tree.to_undirected()
            missing = [n for n in tree.nodes if n not in depths]
            for node in missing:
                min_d = None
                for root in roots:
                    try:
                        d = nx.shortest_path_length(undir, source=root, target=node)
                    except Exception:
                        continue
                    if min_d is None or d < min_d:
                        min_d = d
                depths[node] = int(min_d) if min_d is not None else 0

        depth_values = np.asarray(list(depths.values()), dtype=float)

        branching = []
        for node in tree.nodes:
            children = list(tree.successors(node))
            if len(children) == 0:
                continue

            # Exclude parents whose children are all leaves (last branching level).
            if all(tree.out_degree(child) == 0 for child in children):
                continue

            branching.append(float(len(children)))
    else:
        root = next(iter(tree.nodes))
        depth_map = nx.single_source_shortest_path_length(tree, root)
        depth_values = np.asarray(list(depth_map.values()), dtype=float)

        branching = []
        for node in tree.nodes:
            deg = tree.degree(node)
            if deg == 0:
                continue
            # Undirected fallback: approximate children as neighbors away from root.
            node_depth = depth_map.get(node, 0)
            children = [nbr for nbr in tree.neighbors(node) if depth_map.get(nbr, node_depth) > node_depth]
            if len(children) == 0:
                continue
            if all(tree.degree(child) <= 1 for child in children):
                continue
            branching.append(float(len(children)))

    return {
        "height": float(np.max(depth_values)) if depth_values.size > 0 else None,
        "avg_depth": float(np.mean(depth_values)) if depth_values.size > 0 else None,
        "branching_factor_min": float(np.min(branching)) if len(branching) > 0 else None,
        "branching_factor_max": float(np.max(branching)) if len(branching) > 0 else None,
        "branching_factor_avg": float(np.mean(branching)) if len(branching) > 0 else None,
    }


def _build_hierarchical_caches(tree):
    """Precompute per-node caches needed for hierarchical metrics.

    Parameters
    ----------
    tree : directed networkx DiGraph (the ground-truth taxonomy tree)

    Returns
    -------
    dict with keys:
        root            – root node id
        ancestors_map   – {node: frozenset of proper ancestor node ids}
        leaves_per_node – {node: number of leaf descendants (self counts as 1 if leaf)}
        root_leaf_count – total number of leaves (int)
        log_root        – log(root_leaf_count), pre-computed for coverage formula
    Empty dict is returned when the tree is None, empty, or undirected.
    """
    if tree is None or not hasattr(tree, "is_directed") or not tree.is_directed():
        return {}
    if len(tree.nodes) == 0:
        return {}

    roots = [n for n in tree.nodes if tree.in_degree(n) == 0]
    if not roots:
        return {}
    root = roots[0]

    # Ancestors: all proper ancestors = nodes on the root→node path, excluding self.
    ancestors_map: dict = {}
    for node in tree.nodes:
        try:
            path = nx.shortest_path(tree, root, node)
            ancestors_map[node] = frozenset(path[:-1])
        except nx.NetworkXNoPath:
            ancestors_map[node] = frozenset()

    # Leaf-descendant count per node (leaf = out_degree 0; leaf itself counts as 1).
    leaves_per_node: dict = {}

    def _count_leaves(n: object) -> int:
        children = list(tree.successors(n))
        if not children:
            leaves_per_node[n] = 1
            return 1
        total = sum(_count_leaves(c) for c in children)
        leaves_per_node[n] = total
        return total

    _count_leaves(root)
    # Safety: fill in any node not reachable from root.
    for n in tree.nodes:
        if n not in leaves_per_node:
            leaves_per_node[n] = 1

    root_leaf_count = max(leaves_per_node.get(root, 1), 1)
    log_root = float(np.log(root_leaf_count)) if root_leaf_count > 1 else 1.0

    return {
        "root": root,
        "ancestors_map": ancestors_map,
        "leaves_per_node": leaves_per_node,
        "root_leaf_count": root_leaf_count,
        "log_root": log_root,
    }


def compute_eval_metrics(true_labels, pred_labels, tree=None, gt_tree=None):
    logger = logging.getLogger("recsiam.eval.compute_eval_metrics")

    tree_stats = _hierarchy_stats(tree)
    gt_tree_stats = _hierarchy_stats(gt_tree) if gt_tree is not None else {
        "height": None,
        "avg_depth": None,
        "branching_factor_min": None,
        "branching_factor_max": None,
        "branching_factor_avg": None,
    }

    if len(true_labels) == 0:
        return {
            "accuracy": None,
            "precision": None,
            "f1": None,
            "mean_geodesic_distance": None,
            "hierarchical_accuracy": None,
            "mean_coverage": None,
            "hAURC": None,
            "tree_height": tree_stats["height"],
            "tree_avg_depth": tree_stats["avg_depth"],
            "tree_branching_factor_min": tree_stats["branching_factor_min"],
            "tree_branching_factor_max": tree_stats["branching_factor_max"],
            "tree_branching_factor_avg": tree_stats["branching_factor_avg"],
            "gt_tree_height": gt_tree_stats["height"],
            "gt_tree_avg_depth": gt_tree_stats["avg_depth"],
            "gt_tree_branching_factor_min": gt_tree_stats["branching_factor_min"],
            "gt_tree_branching_factor_max": gt_tree_stats["branching_factor_max"],
            "gt_tree_branching_factor_avg": gt_tree_stats["branching_factor_avg"],
        }

    true_labels = np.asarray(true_labels, dtype=object)
    pred_labels = np.asarray(pred_labels, dtype=object)

    accuracy = float(np.mean(true_labels == pred_labels))
    # Use set + sort-by-str to handle mixed types (str leaves vs int internal nodes
    # when thr_a > 0 causes the model to stop at an internal node).
    labels = np.array(
        sorted(set(true_labels.tolist()) | set(pred_labels.tolist()), key=str),
        dtype=object,
    )

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
    # Use gt_tree for geodesic if provided; fall back to tree.
    geo_tree = gt_tree if gt_tree is not None else tree
    if geo_tree is not None and len(geo_tree.nodes) > 0:
        undir = geo_tree.to_undirected()
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
    elif geo_tree is None:
        logger.info("Geodesic diagnostics: no tree available, geodesic distance disabled for this evaluation")
    else:
        logger.warning("Geodesic diagnostics: evaluation tree is empty, geodesic distance disabled")

    # ------------------------------------------------------------------
    # Hierarchical correctness, coverage, and hAURC
    # Uses geo_tree (ground-truth directed taxonomy) when available.
    # ------------------------------------------------------------------
    hierarchical_accuracy = None
    mean_coverage = None
    hAURC = None

    h_cache = _build_hierarchical_caches(geo_tree)
    if h_cache:
        ancestors_map = h_cache["ancestors_map"]
        leaves_per_node = h_cache["leaves_per_node"]
        log_root = h_cache["log_root"]

        correct_H_list = []
        coverage_list = []

        for t_lab, p_lab in zip(true_labels, pred_labels):
            # Hierarchical correctness: pred == true OR pred is proper ancestor of true.
            is_h_correct = float(
                p_lab == t_lab or p_lab in ancestors_map.get(t_lab, frozenset())
            )
            correct_H_list.append(is_h_correct)

            # Coverage: 1 - log(|Leaves(pred)|) / log(|Leaves(root)|)
            n_leaves = leaves_per_node.get(p_lab, 1)
            cov = 1.0 - float(np.log(max(n_leaves, 1))) / log_root
            coverage_list.append(cov)

        if correct_H_list:
            correct_H_arr = np.array(correct_H_list, dtype=float)
            coverage_arr = np.array(coverage_list, dtype=float)

            hierarchical_accuracy = float(np.mean(correct_H_arr))
            mean_coverage = float(np.mean(coverage_arr))

            # hAURC: sort samples by coverage descending (most specific first),
            # compute the running (mean_coverage, hierarchical_risk) curve,
            # then integrate with the trapezoidal rule.
            # Lower hAURC = low risk even when predictions are highly specific.
            order = np.argsort(coverage_arr)[::-1]
            s_cov = coverage_arr[order]
            s_corr = correct_H_arr[order]
            n = len(s_cov)
            k_idx = np.arange(1, n + 1, dtype=float)
            cum_mean_cov = np.cumsum(s_cov) / k_idx   # running mean coverage (decreasing)
            cum_risk = 1.0 - np.cumsum(s_corr) / k_idx  # running hierarchical risk

            if n >= 2:
                # Flip so coverage is ascending for np.trapz (∫ R_H(c) dc).
                hAURC = float(np.trapz(cum_risk[::-1], cum_mean_cov[::-1]))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "f1": f1,
        "mean_geodesic_distance": mean_geo,
        "hierarchical_accuracy": hierarchical_accuracy,
        "mean_coverage": mean_coverage,
        "hAURC": hAURC,
        "tree_height": tree_stats["height"],
        "tree_avg_depth": tree_stats["avg_depth"],
        "tree_branching_factor_min": tree_stats["branching_factor_min"],
        "tree_branching_factor_max": tree_stats["branching_factor_max"],
        "tree_branching_factor_avg": tree_stats["branching_factor_avg"],
        "gt_tree_height": gt_tree_stats["height"],
        "gt_tree_avg_depth": gt_tree_stats["avg_depth"],
        "gt_tree_branching_factor_min": gt_tree_stats["branching_factor_min"],
        "gt_tree_branching_factor_max": gt_tree_stats["branching_factor_max"],
        "gt_tree_branching_factor_avg": gt_tree_stats["branching_factor_avg"],
    }


def evaluate_test_samples(obj_mem, test_samples, embedding_loader, thr_a=0.0, thr_r=None,
                          logger=None, gt_tree=None, batch_size=200):
    """
    Same behavior and return value as before, but loads embeddings and
    calls obj_mem.predict() in batches instead of once per sample. This
    avoids paying per-call overhead (including any CUDA sync inside
    predict) and serial disk I/O for every single test sample, which is
    what made evaluation slow on large test sets. Each sample can still
    consist of multiple embedding rows (a single file may contain more
    than one vector); majority_vote is still applied per-sample, not
    across the whole batch, so results are identical to the unbatched
    version -- this only changes how we get there.
    """
    if logger is None:
        logger = logging.getLogger("recsiam.eval")

    true_labels = []
    pred_labels = []

    n = len(test_samples)
    for start in range(0, n, batch_size):
        chunk = test_samples[start:start + batch_size]

        # Load this chunk's embeddings, remembering how many rows each
        # sample contributed so we can re-split predictions per-sample.
        chunk_embs = []
        row_counts = []
        chunk_targets = []
        chunk_paths = []
        for sample in chunk:
            try:
                emb = embedding_loader(sample["path"])
            except Exception as exc:
                logger.warning("Skipping test sample '%s' due to load error: %s",
                               sample["path"], exc)
                continue
            chunk_embs.append(emb)
            row_counts.append(len(emb))
            chunk_targets.append(sample["target"])
            chunk_paths.append(sample["path"])

        if not chunk_embs:
            continue

        batch_emb = np.concatenate(chunk_embs, axis=0)

        try:
            pred, _, _ = obj_mem.predict(batch_emb, thr_a=thr_a, thr_r=thr_r)
        except Exception as exc:
            # Fall back to per-sample calls only for this chunk, so a
            # single bad chunk doesn't lose every sample in it -- mirrors
            # the original per-sample try/except behavior.
            logger.warning(
                "Batched prediction failed for chunk starting at %d (%s); "
                "falling back to per-sample prediction for this chunk.",
                start, exc,
            )
            for target, path, emb in zip(chunk_targets, chunk_paths, chunk_embs):
                try:
                    sample_pred_arr, _, _ = obj_mem.predict(emb, thr_a=thr_a, thr_r=thr_r)
                except Exception as exc2:
                    logger.warning("Skipping test sample '%s' due to prediction error: %s",
                                   path, exc2)
                    continue
                sample_pred_arr = np.asarray(sample_pred_arr, dtype=object)
                if sample_pred_arr.ndim == 0:
                    sample_pred_arr = sample_pred_arr[None]
                sample_pred = majority_vote(sample_pred_arr)
                if sample_pred is None:
                    continue
                true_labels.append(target)
                pred_labels.append(sample_pred)
            continue

        pred = np.asarray(pred, dtype=object)
        if pred.ndim == 0:
            pred = pred[None]

        # Re-split the batch predictions back into per-sample groups
        # (in the same order rows were concatenated) and majority-vote
        # each group independently, exactly as the unbatched version did
        # per sample.
        offset = 0
        for target, n_rows in zip(chunk_targets, row_counts):
            sample_pred_rows = pred[offset:offset + n_rows]
            offset += n_rows
            sample_pred = majority_vote(sample_pred_rows)
            if sample_pred is None:
                continue
            true_labels.append(target)
            pred_labels.append(sample_pred)

    return compute_eval_metrics(true_labels, pred_labels, obj_mem.T, gt_tree=gt_tree)