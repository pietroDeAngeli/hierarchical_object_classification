from __future__ import division
import itertools
import logging
import torch
import numpy as np
import networkx as nx

from . import supervision as sup
from . import utils
from .utils import a_app, t_app
from . import evm

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
                 predict_policy="tree"):
        self.evm_args = evm_args

        self.M = np.array([])

        self.T = nx.DiGraph()

        self.inst = {}
        self.sup = {}
        self.elem_ids = np.array([], dtype=object)
        self.elem_node_ids = np.array([], dtype=object)
        assert force in ("top", "bot", None)
        self.force = force

        self.predict_policy = predict_policy
        assert predict_policy in ("tree", "all")

        self.update_policy = update_policy
        self.rng = rng
        self.bootstrap = bootstrap
        self.leaves = 0
        self.logger = logging.getLogger("recsiam.memory.ObjectsMemory")


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

        if not is_leaf or new_genus or len(self.T) == 0:
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

        self.sup[new_id] = supervised

        if self.leaves == self.bootstrap:
            self.recompute_all_update_evm()
        elif self.leaves > self.bootstrap:
            self.update_evm(new_id_node, new_data, pred)

    def new_evm(self):
        return evm.EVM(**self.evm_args)

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

    def recompute_evm_for_node(self, node):

        childs = list(self.T.succ[node])
        labels = np.empty(self.elem_ids.size, dtype=object)
        data_mask = np.zeros(self.elem_ids.size, dtype=np.bool_)

        for c in childs:
            c_elems = self.T.nodes[c]["elem"] | self.T.nodes[c]["uelem"]
            c_mask = utils.arrayinset(self.elem_node_ids, c_elems)

            assert not np.any(data_mask & c_mask)
            data_mask |= c_mask
            labels[c_mask] = c

        labels = labels[data_mask]
        points = self.M[data_mask]
        negatives = self.M[~data_mask]

        self.T.nodes[node]["cls"] = self.new_evm()
        self.T.nodes[node]["cls"].fit(points, labels, neg_X=negatives)

    def recompute_all_update_evm(self):
        for node in self.T.nodes:
            if not utils.is_leaf(self.T, node):
                self.recompute_evm_for_node(node)

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

        cls = self.T.nodes[utils.get_root(self.T)]["cls"]
        ndim = cls.X.ndim
        assert data.ndim - ndim in (0, -1)

        unsq = False
        if data.ndim < ndim:
            unsq = True
            data = data[None, :]

        res = np.array([self.predict_single(d, *args, **kwargs) for d in data], dtype=object)

        if not unsq:
            res = res.T
        else:
            res = np.squeeze(res.T)

        prob = res[2]

        return res[0], res[1], prob

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
