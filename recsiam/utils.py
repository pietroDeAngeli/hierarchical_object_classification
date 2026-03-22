import itertools
import random


import torch
import numpy as np
import sklearn.utils
import networkx as nx


class ImageNormalizer(torch.nn.Module):

    def __init__(self, mean, std, channel_first=True):
        super(ImageNormalizer, self).__init__()
        self.channel_first = channel_first
        if self.channel_first:
            mean = torch.from_numpy(np.asarray(mean)[:, None, None])
            std = torch.from_numpy(np.asarray(std)[:, None, None])
        else:
            mean = torch.from_numpy(np.asarray(mean)[:, None, None])
            std = torch.from_numpy(np.asarray(std)[:, None, None])

        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float())

    def forward(self, batch):
        zero_to_one = batch.float() / 255.
        return (zero_to_one - self.mean) / self.std


# I don't need this for now since I'm using DINO
#def default_image_normalizer():
#    return ImageNormalizer([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def t2a(tensor):
    return tensor.cpu().detach().numpy()


def a2t(array_like):
    return astensor(array_like)


def empty_cat(args):
    not_null = tuple(a for a in args if a.size > 0)
    if len(not_null) == 0:
        return args[0]
    elif len(not_null) == 1:
        return not_null[0]
    else:
        return np.concatenate(not_null)


def assqarray(data):
    if isinstance(data, (np.ndarray, list, tuple)):
        if isinstance(data, np.ndarray):
            sq = data.squeeze()
            if sq.shape == ():
                sq = np.array([sq])
        else:
            raise ValueError("unsupported type {} ".format(type(data)))

    else:
        sq = np.array([data])

    return sq


def as_list(elem):
    if type(elem) == list:
        return elem
    elif isinstance(elem, np.ndarray):
        return list(elem)
    else:
        return [elem]


def astensor(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    else:
        return torch.tensor(data)


def a_app(arr, elem, ndim=1):

    elem_ndim = len(elem.shape)
    if ndim == elem_ndim:
        elem_exp = elem
    else:
        final_dim = (None,) * (ndim - elem_ndim) + (...,)
        elem_exp = elem[final_dim]

    if len(arr) == 0:
        return elem_exp.copy()
    else:
        return np.concatenate([arr, elem_exp])


def t_app(tensor, elem, ndim=1):
    elem_ndim = len(elem.shape)
    if ndim == elem_ndim:
        elem_exp = elem
    else:
        final_dim = (None,) * (ndim - elem_ndim) + (...,)
        elem_exp = elem[final_dim]

    if len(tensor) == 0:
        return elem_exp.clone()
    else:
        return torch.cat([tensor, elem_exp])


def safe_partition(a, k):
    if a.shape[-1] <= k:
        return a
    return np.partition(a, k, axis=-1)[..., :k]


def empty_column_stack(args):
    not_null = tuple(a for a in args if a.size > 0)
    if len(not_null) == 0:
        return args[0]
    elif len(not_null) == 1:
        return not_null[0]
    else:
        return np.column_stack(not_null)


def inverse_argsort(array_like):
    idx = np.arange(len(array_like))
    inverse = np.arange(len(array_like))

    seq_argsorted = np.argsort(array_like)

    inverse[seq_argsorted[idx]] = idx

    return inverse


def shuffle_with_probablity(labels, prob_new, seed):
    rnd = sklearn.utils.check_random_state(seed)


    orig_state = random.getstate()

    random.seed(rnd.randint(2**32 -1))

    uniq_lab = np.unique(labels)

    l_s = rnd.permutation(np.array([set(np.where(labels == l)[0]) for l in uniq_lab]))

    s_old = set()

    final_order = np.tile(-1, len(labels))
    get_new = rnd.uniform(size=len(labels)) < prob_new

    l_s_ind = 0

    for itx in range(len(labels)):

        if (get_new[itx] and l_s_ind < l_s.shape[0]) or len(s_old) == 0:
                new_set = l_s[l_s_ind]

                new_val = random.sample(new_set, 1)[0]
                new_set.remove(new_val)
                s_old.update(new_set)

                l_s_ind += 1

        else:
                new_val = random.sample(s_old, 1)[0]
                s_old.remove(new_val)

        final_order[itx] = new_val


    assert (final_order >= 0).all()

    random.setstate(orig_state)

    return final_order


def tree_with_instances(info, ids, pref):
    tree = tree_from_list(info)
    nodeset = set(tree.nodes)
    for e, i in enumerate(ids):
        assert i in nodeset
        tree.add_edge(i, pref + str(e))

    return tree

def tree_from_list(info, permissive=False):
    tree = nx.DiGraph()

    if type(info) != list or len(info) > 0:
        rec_tree_from_lists(info, tree, None, permissive)
    return tree


def new_node_id(tree):
    return - (len(tree.nodes) + 1)


def parent(tree, node):
    return next(iter(tree.pred[node]))


def rec_tree_from_lists(lists, G, parent, permissive):

    if type(lists) != list:
        node = lists
        elem = set((lists,))
    else:
        #if not permissive:
            #assert len(lists) >= 2
        node = new_node_id(G)
        G.add_node(node)
        elem = [rec_tree_from_lists(child, G, node, permissive)
                for child in lists]
        elem = set().union(*elem)

    G.add_node(node, elem=elem)
    if parent is not None:
        G.add_edge(parent, node)
    return elem


def shuffle_tree_by_distance(rng, hierarchy, classes, prob):
    taken = np.zeros(len(classes), dtype=np.bool_)
    pref = "inst_"
    tree = tree_with_instances(hierarchy, classes, pref)
    depth = len(nx.algorithms.dag.dag_longest_path(tree)) - 1

    # Support relative mode: prob=1 starts from leaves, prob=0 starts from root.
    if np.isscalar(prob):
        if prob not in (0, 1, 0.0, 1.0):
            raise ValueError("Scalar prob must be 0 or 1 in tree setting")
        prob_vec = np.zeros(depth, dtype=float)
        prob_vec[-1 if float(prob) == 1.0 else 0] = 1.0
        prob = prob_vec
    else:
        prob = np.asarray(prob, dtype=float)

    assert prob.sum() == 1.

    assert depth == len(prob)

    order = np.tile(-1, taken.size)

    last = parent(tree, pref + str(rng.choice(taken.size)))
    for i in range(taken.size):
        curr_prob = prob.copy()
        candidates = ()
        while len(candidates) == 0:
            up = rng.choice(curr_prob.size, p=curr_prob)
            anchor = last
            for _ in range(up):
                preds = list(tree.pred[anchor])
                if not preds:
                    break  # already at root, cannot go further up
                anchor = preds[0]

            candidates = sorted(itertools.chain.from_iterable((tree.succ[n] for n in tree.nodes[anchor]["elem"])))
            if len(candidates) == 0:
                if curr_prob[:up + 1].sum() == 1.:
                    curr_prob[:up + 1] = 0
                    curr_prob[up + 1] = 1.0
                else:
                    curr_prob[up] = 0.
                    curr_prob /= curr_prob.sum()

        choosen = rng.choice(candidates)
        choosen_ind = int(choosen.lstrip(pref))
        taken[choosen_ind] = True
        order[i] = choosen_ind

        last = parent(tree, choosen)
        tree.remove_node(choosen)

    assert taken.all()
    return order


def epoch_seed(seed, epoch):
    if seed is None:
        return seed
    rnd = sklearn.utils.check_random_state(seed)
    return rnd.randint(2**32, size=(epoch,))[-1]


def default_notimplemented(*args):
    raise NotImplementedError()


def default_ignore(*args):
    pass


def reduce_packed_array(target, indices):
    res = np.zeros(len(indices), dtype=indices.dtype)
    res[0] = target[:indices[0]].argmin()
    for i in range(1, len(indices)):
        res[i] = target[indices[i - 1]:indices[i]].argmin()
    return res


def rejection_sampling(N, f_sample, f_accept):
    out = f_sample(N)
    mask = f_accept(out)
    reject, = np.where(~mask)
    while reject.size > 0:
        fill = f_sample(reject.size)
        mask = f_accept(fill)
        out[reject[mask]] = fill[mask]
        reject = reject[~mask]
    return out


_EPS = 1e-7


def safe_pow(data, *args, **kwargs):
#    assert not (data == 0.).any()
    data = data + ((data == 0.).float() * _EPS)
    return torch.pow(data, *args, **kwargs)


def cc2clusters(G):

    cl = np.arange(len(G.nodes))
    cc_id = 0
    for cc in nx.connected_components(G):
        for node in cc:
            cl[node] = cc_id

        cc_id += 1

    return cl


def gen_parents(node, tree):
    while len(tree.pred[node]) > 0:
        par = next(iter(tree.pred[node]))
        node = par
        yield par


def is_leaf(tree, node):
    return len(tree.succ[node]) == 0


def get_root(tree):
    if len(tree.nodes) == 0:
        return None
    roots = [n for n, d in tree.in_degree() if d == 0]
    assert len(roots) == 1
    return roots[0]


def is_ancestor(tree, candidate, target):
    return nx.algorithms.shortest_paths.generic.has_path(tree, candidate,
                                                         target)

def _arrayinset(a, s, r):
    for i in range(len(a)):
        r[i] = a[i] in s


def arrayinset(a, s):
    r = np.empty(len(a), dtype=np.bool_)
    _arrayinset(a, s, r)
    return r
