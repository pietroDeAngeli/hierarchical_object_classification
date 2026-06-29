from __future__ import division
import itertools
import logging
import time
import torch
import numpy as np
import networkx as nx
import faiss as _faiss

from . import supervision as sup
from . import utils
from .utils import a_app, t_app
from . import evm
from .prof_utils import timed, report, dump_csv, reset



_set_fields = ("elem", "uelem")


def propagate_id(tree, node, identifier, field):
    for n in itertools.chain((node,), sup.gen_parents(node, tree)):
        tree.nodes[n][field].add(identifier)


def insert_node(tree, child, parent, middle):
    tree.remove_edge(parent, child)
    tree.add_edge(parent, middle)
    tree.add_edge(middle, child)


def insert_genus(tree, child, parent, new):
    genus = -(len(tree.nodes) + 1)
    tree.add_edge(genus, new)
    insert_node(tree, child, parent, genus)
    for sf in _set_fields:
        tree.nodes[genus][sf] = set().union(*[tree.nodes[c][sf] for c in
                                              tree.succ[genus]])

    if tree.nodes[parent]["cls"] is not None:
        tree.nodes[genus]["cls"] = tree.nodes[parent]["cls"].copy()
        cls = tree.nodes[genus]["cls"]
        mask = cls.y != child

        cls.discard(mask)

        parent_mask = tree.nodes[parent]["cls"].y == child
        tree.nodes[parent]["cls"].y[parent_mask] = genus
        tree.nodes[parent]["cls"].update_labels()


def insert_new_root(tree, old_root, new_sibling):
    new_root = -(len(tree.nodes) + 1)
    tree.add_node(new_root, elem=set(), uelem=set(), cls=None)
    tree.add_edge(new_root, old_root)
    tree.add_edge(new_root, new_sibling)
    for sf in _set_fields:
        union = tree.nodes[old_root][sf] | tree.nodes[new_sibling][sf]
        tree.nodes[new_root][sf] = union

    if tree.nodes[old_root]["cls"] is not None:
        new_cls = tree.nodes[old_root]["cls"].copy()
        new_cls.y = np.tile(old_root, len(new_cls.y)).astype(object)

        tree.nodes[new_root]["cls"] = new_cls


class ObjectsMemory(object):

    def __init__(self, evm_args, update_policy="recompute",
                 rng=np.random.RandomState(),
                 bootstrap=2,
                 force=None,
                 predict_policy="tree",
                 evm_batch_size=1,
                 neg_size=None,
                 negative_selection="random"):
        assert negative_selection in ("random", "faiss_siblings"), \
            f"negative_selection must be 'random' or 'faiss_siblings', got {negative_selection!r}"
        self.evm_args = evm_args
        self.neg_size = neg_size
        self.negative_selection = negative_selection

        self.M = np.array([])

        self.T = nx.DiGraph()

        self.inst = {}
        self.sup = {}
        self.elem_ids = np.array([], dtype=object)
        self.elem_node_ids = np.array([], dtype=object)
        self._node_to_indices = {}  # leaf_node_id -> list of sample indices into self.M
        assert force in ("top", "bot", None)
        self.force = force

        self.predict_policy = predict_policy
        assert predict_policy in ("tree", "all")

        self.update_policy = update_policy
        self.rng = rng
        self.bootstrap = bootstrap
        self.leaves = 0
        self.evm_batch_size = max(1, int(evm_batch_size))
        self._pending_batch_count = 0
        self._pending_batch_leaves = set()
        self.logger = logging.getLogger("recsiam.memory.ObjectsMemory")
        self.node_fit_batch_size = 64
        self._faiss_index = None  # optional global FAISS index (set by subclasses)

    def __len__(self):
        return len(self.T.nodes)

    def get_root(self):
        return utils.get_root(self.T)

    def add_element(self, new_id, new_data, target, new_genus, supervised,
                    pred=None):
        assert not (not supervised and  new_genus)
        field = "elem" if supervised else "uelem"

        is_leaf = utils.is_leaf(self.T, target) if target else True
        new_id_node = target

        # Structural changes (new node, new genus, new root) require EVMs
        # to be up-to-date because insert_genus copies and filters the
        # parent's EVM.  Flush any pending batched updates first.
        structural_change = not is_leaf or new_genus or len(self.T) == 0
        if structural_change and self._pending_batch_count > 0:
            self._flush_batch()

        if structural_change:
            self.T.add_node(new_id, elem=set(), uelem=set(), cls=None)
            self.leaves += 1
            new_id_node = new_id

        if not new_genus:
            if not is_leaf:
                self.T.add_edge(target, new_id)

        else:
            if len(self.T.pred[target]) > 0:
                parent = next(iter(self.T.pred[target]))
                insert_genus(self.T, target, parent, new_id)
            else:
                insert_new_root(self.T, target, new_id)

        self.inst[new_id] = new_id_node
        if new_id == new_id_node:
            propagate_id(self.T, new_id_node, new_id, field)

        self.M = a_app(self.M, new_data, ndim=2)
        self.elem_ids = a_app(self.elem_ids,
                              np.tile(new_id, new_data.shape[0]),
                              ndim=1).astype(object)

        self.elem_node_ids = a_app(self.elem_node_ids,
                                   np.tile(new_id_node, new_data.shape[0]),
                                   ndim=1).astype(object)

        # Update reverse index: leaf_node_id -> sample indices
        n_new = new_data.shape[0]
        start_idx = len(self.M) - n_new
        if new_id_node not in self._node_to_indices:
            self._node_to_indices[new_id_node] = []
        self._node_to_indices[new_id_node].extend(range(start_idx, start_idx + n_new))

        self.sup[new_id] = supervised

        if self.leaves == self.bootstrap:
            self.recompute_all_update_evm()
        elif self.leaves > self.bootstrap:
            if structural_change or self.evm_batch_size <= 1:
                # Structural change: must update immediately
                self.update_evm(new_id_node, new_data, pred)
            else:
                # Adding data to existing leaf: safe to batch
                self._pending_batch_count += 1
                self._pending_batch_leaves.add(new_id_node)
                if self._pending_batch_count >= self.evm_batch_size:
                    self._flush_batch()

    def new_evm(self):
        return evm.EVM(**self.evm_args,
                       neg_size=self.neg_size,
                       negative_selection=self.negative_selection,
                       rng=self.rng)

    def _flush_batch(self):
        """Flush pending batched EVM updates."""
        if self._pending_batch_count == 0:
            return
        if self.update_policy == "recompute":
            # Recompute all ancestor nodes of affected leaves
            dirty_nodes = set()
            for leaf in self._pending_batch_leaves:
                if leaf in self.T.nodes:
                    for anc in utils.gen_parents(leaf, self.T):
                        dirty_nodes.add(anc)
            for node in dirty_nodes:
                if not utils.is_leaf(self.T, node):
                    self.recompute_evm_for_node(node)
        else:
            # For non-recompute policies, fall back to full recompute
            self.recompute_all_update_evm()
        self._pending_batch_count = 0
        self._pending_batch_leaves = set()

    def finalize_updates(self):
        """Flush any pending batched EVM updates."""
        self._flush_batch()

    def update_evm(self, target_id, target_data, pred):
        if self. update_policy == "recompute":
            self.minimal_recompute_all_update_evm(target_id)
        elif self. update_policy == "simple":
            self.simple_update_evm(target_id, target_data)
        elif self.update_policy == "gencomplete":
            self.generative_update_evm(target_id, target_data, self.rng)
        elif self.update_policy == "genminimal":
            self.minimal_generative_update_evm(target_id, target_data,
                                               self.rng)
        elif self.update_policy == "predrec":
            self.predict_recompute_all_update_evm(target_id, target_data,
                                                  pred[0])
        else:
            raise ValueError()

    def minimal_generative_update_evm(self, target_id, target_data, rng):

        node_id = self.inst[target_id]
        path = list(reversed([node_id] + list(utils.gen_parents(node_id,
                                                                self.T))))

        all_dists = np.array(())
        for parent, child in zip(path[:-1], path[1:]):
            cls = self.T.nodes[parent]["cls"]

            gen_distances = cls.generate(rng,
                                         points=target_data,
                                         number=cls.tailsize,
                                         label=child)

            distances = cls.get_distance(target_data, exclude_label=child)

            assert gen_distances.shape == (len(target_data), cls.tailsize)
            all_dists = utils.empty_column_stack((all_dists, gen_distances))
            all_dists = utils.safe_partition(all_dists, cls.tailsize)

            cls.fit(target_data, np.tile(child, len(target_data)),
                    neg_d=all_dists)

            all_dists = utils.empty_column_stack((all_dists, distances))

    def generative_update_evm(self, target_id, target_data, rng):

        node_id = self.inst[target_id]
        path = list(reversed([node_id] + list(utils.gen_parents(node_id,
                                                                self.T))))

        all_points = np.array(())
        for parent, child in zip(path[:-1], path[1:]):
            cls = self.T.nodes[parent]["cls"]

            points, labels = cls.generate(rng)

            labels = labels.astype(object)

            all_points = utils.empty_cat((all_points, points[labels != child]))

            cls.fit(target_data, np.tile(child, len(target_data)),
                    neg_X=all_points)
            all_points = utils.empty_cat((all_points, cls.X[cls.y != child]))

    def simple_update_evm(self, target_id, target_data):

        node_id = self.inst[target_id]
        path = list(reversed([node_id] + list(utils.gen_parents(node_id,
                                                                self.T))))

        for parent, child in zip(path[:-1], path[1:]):
            cls = self.T.nodes[parent]["cls"]
            cls.fit(target_data, np.tile(child, len(target_data)))

    def _query_global_faiss(self, pos_X, pos_idxs):
        """For each positive vector in pos_X (global indices pos_idxs) return the
        neg_size nearest-neighbour *distances* from vectors NOT in pos_idxs.

        k is capped at max_neg_multiplier * tailsize, mirroring the cap
        applied on the random path (EVM._max_neg_cap). Without this, k
        scales with neg_size * n_neg_available and can reach tens of
        thousands on large nodes, even though _weibull_fit only ever
        consumes the `tailsize` nearest negatives per positive -- the rest
        would just cost extra FAISS query time and memory for no benefit.

        We fetch k + n_same + 1 neighbours so that, even after discarding all
        n_same same-class hits, at least k genuine negatives remain
        (assuming n_total > k + n_same).

        Queries are batched to keep peak memory bounded (batch × k_fetch × 12 B).
        Returns float32 array of shape (len(pos_X), k).
        """
        n_total = self._faiss_index.ntotal
        n_same = int(len(pos_idxs))
        n_neg_available = max(1, n_total - n_same)
        # neg_size is a fraction in (0,1]: compute absolute count like _subsample_negatives_random
        k_frac = max(1, round(self.neg_size * n_neg_available))
        tailsize = int(self.evm_args.get("tailsize", 25))
        max_neg_multiplier = self.evm_args.get("max_neg_multiplier", None)
        if max_neg_multiplier is not None:
            k_cap = max(1, int(max_neg_multiplier) * tailsize)
            k = min(k_frac, k_cap)
        else:
            k = k_frac
        # Guarantees k negatives after removing all same-class hits
        k_fetch = min(k + n_same + 1, n_total)

        # Boolean mask: True = belongs to positive class (must be excluded)
        is_pos = np.zeros(n_total, dtype=bool)
        is_pos[pos_idxs] = True

        result = np.empty((len(pos_X), k), dtype=np.float32)
        query_batch = getattr(self, "node_fit_batch_size", 64)

        for qi in range(0, len(pos_X), query_batch):
            batch = np.ascontiguousarray(
                pos_X[qi:qi + query_batch], dtype=np.float32
            )
            D_sq, I = self._faiss_index.search(batch, k_fetch)
            for li in range(len(batch)):
                gi = qi + li
                neg_mask = ~is_pos[I[li]]
                neg_dists = D_sq[li][neg_mask][:k]
                have = len(neg_dists)
                if have < k:
                    # Pad with the furthest available distance (or 0 if empty)
                    pad_val = float(neg_dists[-1]) if have > 0 else 0.0
                    neg_dists = np.pad(neg_dists, (0, k - have),
                                       constant_values=pad_val)
                result[gi] = np.sqrt(np.maximum(neg_dists, 0.0))

        return result

    def _leaves_under(self, node):
        """Return the set of leaf descendants of *node* (or {node} if already a leaf)."""
        if utils.is_leaf(self.T, node):
            return {node}
        return {n for n in nx.descendants(self.T, node) if utils.is_leaf(self.T, n)}

    def recompute_evm_for_node(self, node):
        if not getattr(self, "_logged_selection_mode", False):
            self.logger.info(
                "negative_selection=%r  neg_size=%r",
                self.negative_selection,
                self.neg_size,
            )
            self._logged_selection_mode = True

        childs = list(self.T.succ[node])

        child_to_global_idxs = {}
        for c in childs:
            idxs = []
            for leaf_id in self._leaves_under(c):
                idxs.extend(self._node_to_indices.get(leaf_id, []))
            child_to_global_idxs[c] = np.array(idxs, dtype=np.intp)

        populated_childs = [c for c, idxs in child_to_global_idxs.items() if len(idxs) > 0]

        if len(populated_childs) < 2:
            self.T.nodes[node]["cls"] = None
            self.logger.debug(
                "Skipping EVM fit for node %s: populated_childs=%d",
                node,
                len(populated_childs),
            )
            return

        cls = self.new_evm()

        total_points = sum(len(child_to_global_idxs[c]) for c in populated_childs)
        with timed("recompute_evm_for_node",
                   node=str(node),
                   n_childs=len(populated_childs),
                   n_total=total_points):
            for c in populated_childs:
                pos_idxs = child_to_global_idxs[c]
                pos_X = self.M[pos_idxs]
                y_all = np.tile(c, len(pos_X)).astype(object)

                base_batch = getattr(self, "node_fit_batch_size", 64)
                target_batches_per_child = getattr(self, "node_fit_target_batches", 8)
                batch_size = max(base_batch, (len(pos_X) + target_batches_per_child - 1) // target_batches_per_child)

                if (self.negative_selection == "faiss_siblings"
                        and self.neg_size is not None
                        and self._faiss_index is not None):
                    neg_d_all = self._query_global_faiss(pos_X, pos_idxs)
                    for start in range(0, len(pos_X), batch_size):
                        end = min(start + batch_size, len(pos_X))
                        cls.fit(pos_X[start:end], y_all[start:end],
                                neg_d=neg_d_all[start:end])
                else:
                    sibling_parts = [
                        child_to_global_idxs[s]
                        for s in populated_childs
                        if s != c and len(child_to_global_idxs[s]) > 0
                    ]
                    if not sibling_parts:
                        continue
                    sibling_idxs = np.concatenate(sibling_parts).astype(np.intp)
                    neg_X = self.M[sibling_idxs]
                    for start in range(0, len(pos_X), batch_size):
                        end = min(start + batch_size, len(pos_X))
                        cls.fit(pos_X[start:end], y_all[start:end], neg_X=neg_X)

        if len(cls.y) == 0:
            self.T.nodes[node]["cls"] = None
            return

        self.T.nodes[node]["cls"] = cls

    def recompute_all_update_evm(self):
        for node in self.T.nodes:
            if not utils.is_leaf(self.T, node):
                self.recompute_evm_for_node(node)
        report()
        dump_csv("/tmp/evm_fit_profiling.csv")

    def minimal_recompute_all_update_evm(self, target_id):
        node_id = self.inst[target_id]
        assert utils.is_leaf(self.T, node_id)

        for node in utils.gen_parents(node_id, self.T):
            self.recompute_evm_for_node(node)

    def predict_recompute_all_update_evm(self, target_id, target_data, pred_id):
        node_id = self.inst[target_id]
        #pred_id  = self.inst[pred]
        assert utils.is_leaf(self.T, node_id)
        if node_id == pred_id:
            return 

        pred_path = set(utils.gen_parents(pred_id, self.T)) | {pred_id}
        self.logger.debug("pred_path :{}".format(pred_id))
        self.logger.debug("pred_path :{}".format(pred_path))
        self.logger.debug("node_path :{}".format(list(utils.gen_parents(node_id, self.T))))

        first_found = False
        for node in utils.gen_parents(node_id, self.T):
            if not first_found:
                self.recompute_evm_for_node(node)
            if node in pred_path:
                first_found = True

    def predict(self, data, *args, **kwargs):
        if len(self.T.nodes) == 0:
            raise ValueError("memory is empty")
        elif len(self.T.nodes) == 1:
            node = utils.get_root(self.T)
            node = np.tile(node, len(data)) if data.ndim == 2 else node
            prob = np.ones((len(data), 1)) if data.ndim == 2 else 1.0
            ind = np.zeros(len(data), dtype=int) if data.ndim == 2 else 0

            return node, ind, prob

        # cls = self.T.nodes[utils.get_root(self.T)]["cls"]
        # ndim = cls.X.ndim
        # Embeddings are always 2-D arrays (N, D); a single sample is 1-D (D,).
        ndim = 2
        assert data.ndim - ndim in (0, -1)

        unsq = False
        if data.ndim < ndim:
            unsq = True
            data = data[None, :]

        reset()

        if self.predict_policy == "tree" and self.force != "top" and len(data) > 1:
            nodes, indices, probs = self._predict_batch_from_root(data, *args, **kwargs)
            res = np.empty((len(data), 3), dtype=object)
            res[:, 0] = nodes
            res[:, 1] = indices
            res[:, 2] = probs
        else:
            res = np.array([self.predict_single(d, *args, **kwargs) for d in data], dtype=object)

        if not unsq:
            res = res.T
        else:
            res = np.squeeze(res.T)

        prob = res[2]

        return res[0], res[1], prob

    def _predict_batch_from_root(self, data, thr_a, thr_r=None):
        """Batched equivalent of calling predict_single_from_root(d, ...)
        for every row d in data, one at a time.

        Instead of walking the tree row-by-row (which re-pays the cost of
        EVM.predict's distance computation for every single sample even
        though it already supports batched queries), this groups samples
        by their current tree node at each level and queries that node's
        EVM once for the whole group. Stopping/advancing logic per sample
        is identical to predict_single_from_root: a sample advances when
        probabilities[i] >= thr_r, and stops when probabilities[i] < thr_a
        (unless force == "bot"). Results are identical to calling the
        unbatched version row-by-row; only how we get there changes.
        """
        n = len(data)
        if thr_r is None:
            thr_r = thr_a

        root = utils.get_root(self.T)
        current = np.full(n, root, dtype=object)
        index = np.zeros(n, dtype=object)
        prob = np.empty(n, dtype=object)
        prob[:] = [np.ones(1) for _ in range(n)]

        active = np.array([not utils.is_leaf(self.T, root)] * n, dtype=bool)

        while active.any():
            active_idx = np.where(active)[0]
            # Group active samples by their current node so each node's
            # EVM is queried exactly once per level, on all samples that
            # are at that node right now.
            nodes_at_level = current[active_idx]
            for node in np.unique(nodes_at_level):
                mask_in_group = nodes_at_level == node
                rows = active_idx[mask_in_group]

                pred_labels, i_arr, probabilities = self.T.nodes[node]["cls"].predict(
                    data[rows], True
                )
                # probabilities: shape (len(rows), n_children); i_arr: shape (len(rows),)
                chosen_prob = probabilities[np.arange(len(rows)), i_arr]

                advance_mask = (chosen_prob >= thr_r) | (self.force == "bot")
                stop_mask = (chosen_prob < thr_a) & (self.force != "bot")

                for j, row in enumerate(rows):
                    if advance_mask[j]:
                        current[row] = pred_labels[j]
                        prob[row] = probabilities[j]
                        index[row] = i_arr[j]
                    if stop_mask[j] or utils.is_leaf(self.T, current[row]):
                        active[row] = False

        for row in range(n):
            assert index[row] in range(len(prob[row]))

        return current, index, prob

    def predict_single(self, *args, **kwargs):
        if self.predict_policy == "tree":
            return self.predict_single_from_root(*args, **kwargs)
        elif self.predict_policy == "all":
            return self.predict_single_all(*args, **kwargs)
        else:
            raise ValueError()


    def predict_single_from_root(self, data, thr_a, thr_r=None):
        if len(self.T.nodes) == 0:
            raise ValueError("Memory is empty")
        current = utils.get_root(self.T)
        prob = np.ones(1)
        index = 0

        if thr_r is None:
            thr_r = thr_a

        if self.force != "top":
            while not utils.is_leaf(self.T, current):
                pred, i, probabilities = self.T.nodes[current]["cls"].predict(data, True)
                if probabilities[i] >= thr_r or self.force == "bot":
                    current = pred
                    prob = probabilities
                    index = i
                if probabilities[i] < thr_a and self.force != "bot":
                    break

            assert index in range(len(prob))
        return current, index, prob

    def predict_single_all(self, data, thr_a, thr_r=None):
        if len(self.T.nodes) == 0:
            raise ValueError("Memory is empty")

        currents = None
        best = None
        best_prob = 0.0
        index = 0
        prob = 0.0

        while best_prob < thr_a:
            if currents is None:
                currents = [n for n in self.T.nodes if utils.is_leaf(self.T, n)]
            else:
                currents = set([next(iter(self.T.pred[c])) for c in currents])

            for c in currents:
                pred, i, probabilities = self.T.nodes[c]["cls"].predict(data, True)
                if probabilities[i] >= best_prob:
                    best = pred
                    best_prob = probabilities[i]
                    prob = probabilities
                    index = i

        return best, index, prob

class StaticHierarchyMemory(ObjectsMemory):
    """ObjectsMemory variant with fixed topology.

    This class disables structural changes of the hierarchy while preserving the
    same EVM update logic used by the online module.
    """

    def __init__(self, hierarchy, evm_args, num_elements, update_policy="recompute",
                 rng=np.random.RandomState(), bootstrap=2, force=None,
                 predict_policy="tree", evm_batch_size=1, neg_size=None,
                 negative_selection="random", max_positives_per_node=20000 ):
        if evm_batch_size < 1:
            raise ValueError("evm_batch_size must be >= 1")
        super().__init__(evm_args=evm_args,
                         update_policy=update_policy,
                         rng=rng,
                         bootstrap=bootstrap,
                         force=force,
                         predict_policy=predict_policy,
                         neg_size=neg_size,
                         negative_selection=negative_selection)
        self.T = _build_fixed_tree(hierarchy)
        self.evm_batch_size = int(evm_batch_size)
        # Cap on positives per node before stratified subsampling kicks in
        # (static scenario only). None disables subsampling entirely.
        self.max_positives_per_node = max_positives_per_node
        self._pending_updates = []
        self._pending_leaf_updates = set()
        self._pending_update_count = 0
        self._leaf_nodes = set(n for n in self.T.nodes if utils.is_leaf(self.T, n))
        self._num_elements = int(num_elements)
        self._total_samples: int = 0  # write cursor / count of inserted rows

    def recompute_evm_for_node(self, node):
        """Static-memory override of the base recompute.

        Identical to ObjectsMemory.recompute_evm_for_node, except for an
        added stratified positive-subsampling step (see below) used to
        isolate its effect on accuracy. No d_old reuse, no negative cap,
        no negative-subsample-outside-loop change: those are intentionally
        NOT included here so this override tests positive sampling alone.
        """
        if not getattr(self, "_logged_selection_mode", False):
            self.logger.info(
                "negative_selection=%r  neg_size=%r",
                self.negative_selection,
                self.neg_size,
            )
            self._logged_selection_mode = True

        childs = list(self.T.succ[node])

        child_to_global_idxs = {}
        for c in childs:
            idxs = []
            for leaf_id in self._leaves_under(c):
                idxs.extend(self._node_to_indices.get(leaf_id, []))
            child_to_global_idxs[c] = np.array(idxs, dtype=np.intp)

        populated_childs = [c for c, idxs in child_to_global_idxs.items() if len(idxs) > 0]

        if len(populated_childs) < 2:
            self.T.nodes[node]["cls"] = None
            self.logger.debug(
                "Skipping EVM fit for node %s: populated_childs=%d",
                node,
                len(populated_childs),
            )
            return

        # --- Positive subsampling (static-memory only) ---------------------
        # On nodes high in the tree, child_to_global_idxs aggregates the
        # positives of every descendant, so n_pos can reach hundreds of
        # thousands. The set-cover/reduce step builds a dense n_pos x n_pos
        # distance matrix, whose size (n_pos^2 * 8 bytes) is what drives the
        # process OOM regardless of how the matrix is computed. When the
        # total exceeds max_positives_per_node we subsample the positives,
        # stratified per child: each child keeps a share proportional to its
        # size, but never fewer than a small floor, so small children are
        # not wiped out. The set-cover then still selects the boundary
        # extreme vectors from within the retained subset.
        #
        # NOTE: this is a deliberate approximation of the baseline, active
        # only above the threshold and only in the static scenario.
        max_positives = getattr(self, "max_positives_per_node", 20000)
        total_points = sum(len(child_to_global_idxs[c]) for c in populated_childs)

        if max_positives is not None and total_points > max_positives:
            tailsize = int(self.evm_args.get("tailsize", 25))
            # Floor so each child keeps enough points for a meaningful
            # Weibull tail (but never more than it actually has).
            per_child_floor = max(2, tailsize)

            # First pass: give every child its floor (capped at its size).
            alloc = {}
            for c in populated_childs:
                n_c = len(child_to_global_idxs[c])
                alloc[c] = min(n_c, per_child_floor)

            floored = sum(alloc.values())
            remaining_budget = max_positives - floored

            if remaining_budget <= 0:
                # Even the floors exceed the budget (a node with very many
                # children). Fall back to an even split, still >=1 per child.
                even = max(1, max_positives // len(populated_childs))
                for c in populated_childs:
                    alloc[c] = min(len(child_to_global_idxs[c]), even)
            else:
                # Distribute the remaining budget proportionally to the
                # leftover size of each child (size minus its floor).
                leftovers = {
                    c: len(child_to_global_idxs[c]) - alloc[c]
                    for c in populated_childs
                }
                total_leftover = sum(leftovers.values())
                if total_leftover > 0:
                    for c in populated_childs:
                        extra = int(round(
                            remaining_budget * leftovers[c] / total_leftover
                        ))
                        extra = min(extra, leftovers[c])
                        alloc[c] += extra

            # Draw the per-child samples (sorted for deterministic,
            # memory-friendly indexing into self.M).
            new_child_to_global_idxs = {}
            for c in populated_childs:
                idxs = child_to_global_idxs[c]
                k = min(alloc[c], len(idxs))
                if k >= len(idxs):
                    new_child_to_global_idxs[c] = idxs
                else:
                    chosen = self.rng.choice(len(idxs), size=k, replace=False)
                    new_child_to_global_idxs[c] = idxs[np.sort(chosen)]
            child_to_global_idxs = new_child_to_global_idxs

            new_total = sum(len(child_to_global_idxs[c]) for c in populated_childs)
            self.logger.info(
                "Node %s: positives subsampled %d -> %d "
                "(max_positives_per_node=%d, stratified per child)",
                node, total_points, new_total, max_positives,
            )

        cls = self.new_evm()

        total_points = sum(len(child_to_global_idxs[c]) for c in populated_childs)
        with timed("recompute_evm_for_node",
                   node=str(node),
                   n_childs=len(populated_childs),
                   n_total=total_points):
            for c in populated_childs:
                pos_idxs = child_to_global_idxs[c]
                pos_X = self.M[pos_idxs]
                y_all = np.tile(c, len(pos_X)).astype(object)

                base_batch = getattr(self, "node_fit_batch_size", 64)
                target_batches_per_child = getattr(self, "node_fit_target_batches", 8)
                batch_size = max(base_batch, (len(pos_X) + target_batches_per_child - 1) // target_batches_per_child)

                if (self.negative_selection == "faiss_siblings"
                        and self.neg_size is not None
                        and self._faiss_index is not None):
                    neg_d_all = self._query_global_faiss(pos_X, pos_idxs)
                    for start in range(0, len(pos_X), batch_size):
                        end = min(start + batch_size, len(pos_X))
                        cls.fit(pos_X[start:end], y_all[start:end],
                                neg_d=neg_d_all[start:end])
                else:
                    sibling_parts = [
                        child_to_global_idxs[s]
                        for s in populated_childs
                        if s != c and len(child_to_global_idxs[s]) > 0
                    ]
                    if not sibling_parts:
                        continue
                    sibling_idxs = np.concatenate(sibling_parts).astype(np.intp)
                    neg_X = self.M[sibling_idxs]
                    for start in range(0, len(pos_X), batch_size):
                        end = min(start + batch_size, len(pos_X))
                        cls.fit(pos_X[start:end], y_all[start:end], neg_X=neg_X)

        if len(cls.y) == 0:
            self.T.nodes[node]["cls"] = None
            return

        self.T.nodes[node]["cls"] = cls

    _IVF_NPROBE = 64    # cells searched at query time (~6 % → ~16× speedup, ~98 % recall)
    _IVF_MIN_TRAIN_FLOOR = 2000  # below this many vectors, flat search is already fast; skip IVF
    _IVF_POINTS_PER_CENTROID = 39  # FAISS's own minimum training rule

    def _ivf_nlist_for(self, n):
        """Pick nlist adaptively so IVF activates (and trains validly) on
        datasets of any size, instead of a single fixed nlist that only
        makes sense for one particular dataset size. Standard FAISS rule
        of thumb: nlist ~ sqrt(n), capped so n >= nlist * _IVF_POINTS_PER_CENTROID
        (FAISS's own minimum-training-points requirement) is always met.
        """
        nlist_sqrt_rule = max(1, int(np.sqrt(n)))
        nlist_max_valid = max(1, n // self._IVF_POINTS_PER_CENTROID)
        return max(1, min(nlist_sqrt_rule, nlist_max_valid))

    def _build_global_faiss_index(self):
        """Incrementally update the global FAISS index.

        On first call the index is created:
          - If we have enough vectors (>= _IVF_MIN_TRAIN_FLOOR) an IVFFlat
            index is trained and populated, with nlist chosen adaptively
            for the dataset size (see _ivf_nlist_for) — query time
            substantially faster than flat at ~98 % recall.
          - Otherwise a plain IndexFlatL2 is used (dataset is small enough
            that flat search is already fast, and/or too small for IVF
            training to be statistically meaningful).
        On subsequent calls only the new vectors are appended via .add(); no
        retraining is needed.
        Invariant: self._faiss_index.ntotal == self._total_samples, so FAISS
        position i corresponds to self.M[i] (required by _query_global_faiss).
        """
        n = self._total_samples
        if n == 0:
            return
        if self._faiss_index is None:
            vecs = np.ascontiguousarray(self.M[:n], dtype=np.float32)
            dim = vecs.shape[1]
            if n >= self._IVF_MIN_TRAIN_FLOOR:
                nlist = self._ivf_nlist_for(n)
                quantizer = _faiss.IndexFlatL2(dim)
                index = _faiss.IndexIVFFlat(quantizer, dim, nlist,
                                            _faiss.METRIC_L2)
                index.train(vecs)
                index.nprobe = min(self._IVF_NPROBE, nlist)
                index.add(vecs)
                self.logger.info(
                    "Built IVF FAISS index: %d vectors, dim=%d, "
                    "nlist=%d, nprobe=%d",
                    n, dim, nlist, index.nprobe,
                )
            else:
                index = _faiss.IndexFlatL2(dim)
                index.add(vecs)
                self.logger.info(
                    "Built flat FAISS index: %d vectors, dim=%d "
                    "(< %d vectors needed for IVF)",
                    n, dim, self._IVF_MIN_TRAIN_FLOOR,
                )
            self._faiss_index = index
        else:
            prev = self._faiss_index.ntotal
            if n > prev:
                new_vecs = np.ascontiguousarray(self.M[prev:n], dtype=np.float32)
                self._faiss_index.add(new_vecs)
                self.logger.info(
                    "Updated FAISS index: +%d vectors (total=%d)",
                    n - prev, n,
                )

    def _clear_pending_updates(self):
        self._pending_updates = []
        self._pending_leaf_updates = set()
        self._pending_update_count = 0

    def _flush_pending_updates(self):
        if self._pending_update_count == 0:
            return

        if self.update_policy == "recompute" and self.negative_selection == "faiss_siblings":
            self._build_global_faiss_index()

        if self.update_policy == "recompute":
            dirty_nodes = self.affected_internal_nodes_from_leaves(self._pending_leaf_updates)
            self.logger.info(
                "Flushing %d pending samples touching %d leaves → %d internal nodes to refit",
                self._pending_update_count,
                len(self._pending_leaf_updates),
                len([n for n in dirty_nodes if not utils.is_leaf(self.T, n)]),
            )
            self.recompute_nodes(dirty_nodes)
        else:
            for upd_id, upd_data, upd_pred in self._pending_updates:
                self.update_evm(upd_id, upd_data, upd_pred)

        self._clear_pending_updates()
    
    def _node_depth(self, node):
        root = self.get_root()
        return nx.shortest_path_length(self.T, source=root, target=node)

    def _ordered_internal_nodes(self, nodes, reverse=True):
        internal_nodes = [n for n in nodes if not utils.is_leaf(self.T, n)]
        return sorted(internal_nodes,
                      key=self._node_depth,
                      reverse=reverse)

    def add_element(self, new_id, new_data, target, new_genus, supervised, pred=None):
        if new_genus:
            raise ValueError("StaticHierarchyMemory forbids new genus insertion")
        if target not in self.T.nodes:
            raise KeyError("Target node '{}' is not in the fixed hierarchy".format(target))
        if target not in self._leaf_nodes:
            raise ValueError("Target node '{}' must be a leaf in fixed hierarchy".format(target))

        field = "elem"

        # In static mode each new sample is attached to an existing leaf.
        self.inst[new_id] = target
        propagate_id(self.T, target, new_id, field)

        n_new = new_data.shape[0]
        # Lazy pre-allocation on first call (embedding dim inferred from data).
        if self._total_samples == 0:
            dim = new_data.shape[1]
            self.M = np.empty((self._num_elements, dim), dtype=np.float32)
            self.elem_ids = np.empty(self._num_elements, dtype=object)
            self.elem_node_ids = np.empty(self._num_elements, dtype=object)

        start_idx = self._total_samples
        self._total_samples += n_new
        self.M[start_idx:start_idx + n_new] = new_data.astype(np.float32, copy=False)
        self.elem_ids[start_idx:start_idx + n_new] = np.tile(new_id, n_new)
        self.elem_node_ids[start_idx:start_idx + n_new] = np.tile(target, n_new)

        self.sup[new_id] = supervised

        # Maintain reverse index using the running sample offset.
        if target not in self._node_to_indices:
            self._node_to_indices[target] = []
        self._node_to_indices[target].extend(range(start_idx, start_idx + n_new))

        self._pending_update_count += n_new
        if self.update_policy == "recompute":
            self._pending_leaf_updates.add(target)
        else:
            self._pending_updates.append((new_id, new_data, pred))
        if self._pending_update_count >= self.evm_batch_size:
            self._flush_pending_updates()

    def finalize_updates(self):
        """Flush any pending batched EVM updates and trim pre-allocated arrays."""
        if self._pending_update_count > 0:
            self._flush_pending_updates()
        # Trim in case num_elements was slightly larger than actual insertions
        n = self._total_samples
        if n > 0 and n < len(self.M):
            self.M = self.M[:n]
            self.elem_ids = self.elem_ids[:n]
            self.elem_node_ids = self.elem_node_ids[:n]

    def affected_internal_nodes_from_leaves(self, leaf_nodes):
        dirty_nodes = set()
        for leaf in leaf_nodes:
            dirty_nodes.update(utils.gen_parents(leaf, self.T))
        return dirty_nodes

    def recompute_nodes(self, nodes):
        ordered = self._ordered_internal_nodes(nodes, reverse=True)
        total = len(ordered)
        if total == 0:
            return
        self.logger.info("EVM node fitting: %d internal nodes to process", total)
        t_start = time.time()
        for i, node in enumerate(ordered, 1):
            t_node = time.time()
            self.recompute_evm_for_node(node)
            node_elapsed = time.time() - t_node
            total_elapsed = time.time() - t_start
            avg = total_elapsed / i
            eta = avg * (total - i)
            self.logger.info(
                "EVM node %d/%d  node=%s  node_time=%.1fs  elapsed=%.0fs  ETA=%.0fs",
                i, total, node, node_elapsed, total_elapsed, eta,
            )
        self.logger.info(
            "EVM node fitting complete: %d nodes in %.1fs", total, time.time() - t_start
        )

    def predict_single_from_root(self, data, thr_a, thr_r=None):
        if len(self.T.nodes) == 0:
            raise ValueError("Memory is empty")
        current = utils.get_root(self.T)
        prob = np.ones(1)
        index = 0

        if thr_r is None:
            thr_r = thr_a

        if self.force != "top":
            while not utils.is_leaf(self.T, current):
                cls = self.T.nodes[current]["cls"]
                if cls is None:
                    # No EVM for this node: sparse subtree with only one populated child.
                    # Auto-descend to that child, or pick first populated child if ambiguous.
                    # Loop condition ensures we eventually reach a leaf with string synset name.
                    children = list(self.T.succ[current])
                    populated = [c for c in children
                                 if self.T.nodes[c].get("elem")]
                    if len(populated) >= 1:
                        current = populated[0]  # Pick first populated child
                        continue
                    break
                pred, i, probabilities = cls.predict(data, True)
                if probabilities[i] >= thr_r or self.force == "bot":
                    current = pred
                    prob = probabilities
                    index = i
                if probabilities[i] < thr_a and self.force != "bot":
                    break

            assert index in range(len(prob))
        return current, index, prob

def _build_fixed_tree(hierarchy):
    tree = utils.tree_from_list(hierarchy)
    for node in tree.nodes:
        tree.nodes[node].setdefault("elem", set())
        tree.nodes[node].setdefault("uelem", set())
        tree.nodes[node].setdefault("cls", None)
    return tree


class SupervisionMemory(torch.utils.data.Dataset):

    def __init__(self):
        self.labels = np.array([], dtype=np.int32)
        self.distances = np.array([])

        self.ins_cnt = 0
        self.insertion_orders = np.array([])

    def __len__(self):
        return len(self.distances)

    def __getitem__(self, item):
        return (self.labels[item], self.distances[item])

    def add_entry(self, labels, distance):
        pos = np.searchsorted(self.distances, distance)

        assert np.asarray(distance).size == np.asarray(labels).size

        self.labels = np.insert(self.labels, pos, labels, axis=0)
        self.distances = np.insert(self.distances, pos, distance, axis=0)

        self.insertion_orders = np.insert(self.insertion_orders, pos,
                                          self.ins_cnt, axis=0)
        self.ins_cnt += 1

    def del_entry(self, pos=None):

        if pos is None:
            pos = np.argmin(self.insertion_orders)

        self.labels = np.delete(self.labels, pos, axis=0)
        self.distances = np.delete(self.distances, pos, axis=0)

        self.insertion_orders = np.delete(self.insertion_orders, pos, axis=0)


def compute_linear_threshold(gt, dgt):
    t_cs = gt[::-1].cumsum()[::-1] + (~ gt).cumsum()

    t_indexes = np.where(t_cs == t_cs.max())[0]

    t_ind = t_indexes[len(t_indexes) // 2]

    overflowing = ((t_ind == 0) and not gt[t_ind]) or \
                  ((t_ind == (len(t_cs) - 1)) and gt[t_ind])

    if not overflowing:
        other_ind = t_ind - (gt[t_ind]*2) + 1
        threshold = (dgt[t_ind] + dgt[other_ind]) / 2.0

    else:
        threshold = dgt[t_ind] / 1.05 if t_ind == 0 else dgt[t_ind] * 1.05

    return threshold


def compute_thresolds_from_indexes(gt, dgt, indexes, w_sz):
    l_ind = indexes[len(indexes) // 2]
    u_ind = l_ind + w_sz - 1

    assert l_ind >= 0
    assert u_ind <= len(dgt) - 1

    l_thr = dgt[l_ind-1:l_ind+1].mean() if l_ind > 0 else dgt[l_ind] / 2.
    u_thr = dgt[u_ind:u_ind+2].mean() if u_ind < len(dgt) - 1 else dgt[u_ind] * 2.

    return l_thr, u_thr, l_ind, u_ind


def binary_entropy(p):
    eps = 1e-7
    corr_p = p + np.where(p < eps, eps, 0)
    corr_p = corr_p - np.where(corr_p > (1 - eps), eps, 0)
    p = corr_p
    entropy =  -( p * np.log2(p + eps) + (1 - p) * np.log2(1 - p + eps)  )

    return entropy

def _compute_subtract_entropy_thresholds(gt, dgt, w_sz):

    gt = gt.astype(np.bool_)
    c_win = np.ones(w_sz)

    ara = np.arange(1, gt.size + 1, dtype=np.float64)
    w_ent = binary_entropy(np.convolve(gt, c_win, mode='valid') / w_sz)

    eps_ent = binary_entropy(0.0)
    lb_entropy = binary_entropy(gt[:-w_sz].cumsum() / ara[:-w_sz])
    lb_entropy = np.insert(lb_entropy, 0, eps_ent)

    ub_entropy = binary_entropy((~ gt[w_sz:])[::-1].cumsum() / ara[:-w_sz] )[::-1]
    ub_entropy = np.append(ub_entropy, eps_ent)

    w_div_b = w_ent - lb_entropy - ub_entropy

    indexes = np.where(w_div_b == w_div_b.max())[0]

    res = compute_thresolds_from_indexes(gt, dgt, indexes, w_sz)

    return res +  (w_div_b[res[2]],)


def compute_subtract_entropy_thresholds(gt, dgt, fraction):
    w_sz = max(np.round(len(gt) / fraction**(-1)).astype(int), 1)

    return _compute_subtract_entropy_thresholds(gt, dgt, w_sz)