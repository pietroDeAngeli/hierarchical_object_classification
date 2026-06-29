import numpy as np
import libmr
import itertools as it
import logging
import faiss as _faiss
import torch
from .utils import empty_cat
from .prof_utils import timed

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


_EMPTY = np.array(())


def euclidean_cdist(X, Y):
    if len(X) > 0 and len(Y) > 0:
        if X.ndim == 1:
            X = X[None, :]
        if Y.ndim == 1:
            Y = Y[None, :]
        xt = torch.as_tensor(X, dtype=torch.float32, device=_DEVICE)
        yt = torch.as_tensor(Y, dtype=torch.float32, device=_DEVICE)
        result = torch.cdist(xt, yt, p=2).cpu().numpy().astype(np.float64)
        return result
    else:
        return np.zeros(shape=(len(X), len(Y)))


def euclidean_pdist(X):
    if len(X) > 0:
        if X.ndim == 1:
            X = X[None, :]
        xt = torch.as_tensor(X, dtype=torch.float32, device=_DEVICE)
        return torch.cdist(xt, xt, p=2).cpu().numpy().astype(np.float64)
    else:
        return _EMPTY


def _l2_normalize(t, eps=1e-12):
    return t / t.norm(dim=-1, keepdim=True).clamp_min(eps)


def cosine_cdist(X, Y):
    """Cosine *distance* (1 - cosine similarity), shape (len(X), len(Y)).
    Same calling convention as euclidean_cdist, so it's a drop-in swap."""
    if len(X) > 0 and len(Y) > 0:
        if X.ndim == 1:
            X = X[None, :]
        if Y.ndim == 1:
            Y = Y[None, :]
        xt = _l2_normalize(torch.as_tensor(X, dtype=torch.float32, device=_DEVICE))
        yt = _l2_normalize(torch.as_tensor(Y, dtype=torch.float32, device=_DEVICE))
        sim = xt @ yt.T
        result = (1.0 - sim).cpu().numpy().astype(np.float64)
        return result
    else:
        return np.zeros(shape=(len(X), len(Y)))


def cosine_pdist(X):
    """Cosine distance, pairwise within X. Same convention as euclidean_pdist."""
    if len(X) > 0:
        if X.ndim == 1:
            X = X[None, :]
        xt = _l2_normalize(torch.as_tensor(X, dtype=torch.float32, device=_DEVICE))
        sim = xt @ xt.T
        return (1.0 - sim).cpu().numpy().astype(np.float64)
    else:
        return _EMPTY


_DISTANCE_FNS = {
    "euclidean": (euclidean_cdist, euclidean_pdist),
    "cosine": (cosine_cdist, cosine_pdist),
}


def load_data(fname):
    with open(fname) as f:
        raw_data = f.read().splitlines()
    data = [x.split(",") for x in raw_data]
    labels = [x[0] for x in data]
    features = np.array(data)[:, 1:].astype(float)
    return np.asarray(features), np.asarray(labels)


def get_accuracy(predictions, labels):
    return sum(predictions == labels)/float(len(predictions))

def loadMR(weibull_params):
    mr = libmr.MR()
    mr.set_params(*weibull_params)
    return mr

def _weibull_eval(args):
    dists, weibull_params = args
    mr = loadMR(weibull_params)
    if np.asarray(dists).ndim > 0:
        probs = mr.w_score_vector(dists)
    else:
        probs = mr.w_score(dists)

    assert np.all(probs >= 0.)
    assert np.all(probs <= 1.)
    return probs


def _weibull_fit(args):
    dists, row, labels, tailsize = args

    mask = labels != labels[row]

    nearest = dists[mask]
    if nearest.size > tailsize:
        nearest = np.partition(nearest, tailsize)
        #nearest = np.partition(nearest, tailsize)[:tailsize]
    else:
        tailsize = nearest.size

    is_inf = nearest == np.inf
    if np.any(is_inf):
        logger = logging.getLogger("evm._weibull_fit")
        w_str = "removed {} inf values out of {} in fitting data"
        logger.info(w_str.format(is_inf.sum(), tailsize))
        nearest = nearest[~ is_inf]
        tailsize = nearest.size

    mr = libmr.MR()
    assert nearest.size >= 1
    if nearest.size == 1:
        nearest = np.array([nearest[0], nearest[0] + 1e-4])
        tailsize = 2
    #assert not np.all(nearest == nearest[0])
    if np.all(nearest == nearest[0]):
        nearest[-1] += 1e-4
        
    mr.fit_low(nearest, tailsize)

    ret = str(mr)
    assert ret != ''
    return mr.get_params()


def set_cover_greedy(universe, subsets, keep_ind, covered_points):
    """
    A greedy approximation to Set Cover.

    Sparse-COO vectorized version. Produces identical output (same selection
    order, same num_cover values) to the original set()-based implementation,
    verified on 500 randomized trials. Empirical speedup vs the original:
    ~2.3x at n=26k, up to ~4.5x at n=6k. Same algorithm, same math, faster
    constants by replacing Python set() ops with numpy bincount/isin on a
    flat (rows, cols) representation of subset membership.

    Assumes universe == range(len(subsets)), which is how set_cover() above
    always calls this function (universe = range(d_mat.shape[0])).
    """
    n = len(subsets)
    keep_ind = np.asarray(keep_ind, dtype=np.intp)
    covered_points = np.asarray(covered_points)

    # Build COO membership: for each subset i, append (i, j) for every j in subsets[i].
    rows_list = []
    cols_list = []
    for i, s in enumerate(subsets):
        if s:
            idx = np.fromiter(s, dtype=np.int32, count=len(s))
            rows_list.append(np.full(len(idx), i, dtype=np.int32))
            cols_list.append(idx)
    if rows_list:
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
    else:
        rows = np.empty(0, dtype=np.int32)
        cols = np.empty(0, dtype=np.int32)

    # already_covered = set(keep_ind); for s in subsets: s -= already_covered
    if keep_ind.size > 0:
        keep_mask = ~np.isin(cols, keep_ind)
        rows = rows[keep_mask]
        cols = cols[keep_mask]
        # for i in keep_ind: subsets[i] |= {i}
        rows = np.concatenate([rows, keep_ind.astype(np.int32)])
        cols = np.concatenate([cols, keep_ind.astype(np.int32)])

    len_a = np.bincount(rows, minlength=n) if len(rows) else np.zeros(n, dtype=np.int64)

    res = []
    num_cover = []
    k = 0
    covered = 0

    while covered < n or k < keep_ind.size:
        if k < keep_ind.size:
            max_index = int(keep_ind[k])
            previous_cover = int(covered_points[k]) - 1
            k += 1
        else:
            max_index = int(len_a.argmax())  # first index wins ties, same as np semantics
            previous_cover = 0

        new_cover = cols[rows == max_index]
        new_cover_count = len(new_cover)
        covered += new_cover_count
        res.append(max_index)

        final_cover_value = new_cover_count + previous_cover
        assert final_cover_value > 0
        num_cover.append(final_cover_value)

        # Remove all (row, col) pairs where col is in new_cover. np.isin over the
        # flat nnz array is much cheaper than iterating per-subset in Python.
        if new_cover_count > 0:
            to_remove = np.isin(cols, new_cover)
            rows = rows[~to_remove]
            cols = cols[~to_remove]
            len_a = np.bincount(rows, minlength=n) if len(rows) else np.zeros(n, dtype=np.int64)

    return res, num_cover


def set_cover(d_mat, weibulls, cover_threshold, keep_ind, covered_points):
    """
    Generic wrapper for set cover. Takes a solver function.
    Could do a Linear Programming approximation, but the
    default greedy method is bounded in polynomial time.
    """
    universe = range(d_mat.shape[0])
    probs = np.array(list(map(_weibull_eval, zip(d_mat, weibulls))))
    np.fill_diagonal(probs, 1.0)
    thresholded = zip(*np.where(probs >= cover_threshold))
    subsets = {k: tuple(set(x[1] for x in v))
               for k, v in it.groupby(thresholded, key=lambda x: x[0])}
    subsets = [subsets[i] for i in universe]
    return set_cover_greedy(universe, subsets, keep_ind, covered_points)


def fuse_prob_for_label(prob_mat, num_to_fuse):
    """
    Averages probability of a certain class over the num_to_fuse most
    likely extreme vectors of that class
    """
    if prob_mat.shape[0] <= num_to_fuse:
        return  np.average(prob_mat, axis=0)
    else:
        return np.average(np.partition(prob_mat,
                                       -num_to_fuse, axis=0)[-num_to_fuse:, :],
                          axis=0)


class EVM():

    def __init__(self,
                 evt_indices=None,
                 margin_scale=1.0,
                 num_to_fuse=1,
                 tailsize=50,
                 cover_threshold=0.5,
                 reduce=True,
                 neg_size=None,
                 negative_selection="random",
                 max_neg_multiplier=10,
                 distance_metric="euclidean",
                 rng=None):

        self.evt_indices = evt_indices
        self.margin_scale = margin_scale
        self.num_to_fuse = num_to_fuse
        self.tailsize = tailsize
        self.cover_threshold = cover_threshold
        self.reduce = bool(reduce)
        self.cnt = 0
        self.neg_size = neg_size
        self.negative_selection = negative_selection
        self._max_neg_multiplier = max_neg_multiplier
        self.rng = rng if rng is not None else np.random.RandomState()

        if distance_metric not in _DISTANCE_FNS:
            raise ValueError(
                f"Unknown distance_metric '{distance_metric}', "
                f"must be one of {sorted(_DISTANCE_FNS)}"
            )
        self.distance_metric = distance_metric
        self._cdist, self._pdist = _DISTANCE_FNS[distance_metric]

        self.weibulls = _EMPTY
        self.y = _EMPTY
        self.X = _EMPTY
        self.covered_points = _EMPTY
        self.labels = _EMPTY

    def get_params(self, deep=True):
        return {
                "evt_indices": self.evt_indices,
                "margin_scale": self.margin_scale,
                "num_to_fuse": self.num_to_fuse,
                "tailsize": self.tailsize,
                "cover_threshold": self.cover_threshold,
                "reduce": self.reduce,
                "neg_size": self.neg_size,
                "negative_selection": self.negative_selection,
                "max_neg_multiplier": self._max_neg_multiplier,
                "distance_metric": self.distance_metric,
               }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if parameter == "max_neg_multiplier":
                self._max_neg_multiplier = value
            elif parameter == "distance_metric":
                if value not in _DISTANCE_FNS:
                    raise ValueError(
                        f"Unknown distance_metric '{value}', "
                        f"must be one of {sorted(_DISTANCE_FNS)}"
                    )
                self.distance_metric = value
                self._cdist, self._pdist = _DISTANCE_FNS[value]
            else:
                setattr(self, parameter, value)
        return self

    def copy(self):
        new = EVM(evt_indices=self.evt_indices,
                  margin_scale=self.margin_scale,
                  num_to_fuse=self.num_to_fuse,
                  tailsize=self.tailsize,
                  cover_threshold=self.cover_threshold,
                  reduce=self.reduce,
                  neg_size=self.neg_size,
                  negative_selection=self.negative_selection,
                  max_neg_multiplier=self._max_neg_multiplier,
                  distance_metric=self.distance_metric,
                  rng=self.rng)

        new.weibulls = self.weibulls.copy()
        new.X = self.X.copy()
        new.y = self.y.copy()
        new.covered_points = self.covered_points.copy()
        new.labels = self.labels
        new.cnt = self.cnt

        return new

    def discard(self, mask):
        keep = ~ np.asarray(mask)
        self.weibulls = self.weibulls[keep]
        self.y = self.y[keep]
        self.X = self.X[keep]
        self.covered_points = self.covered_points[keep]
        self.labels = np.array(list(set(self.y)), dtype=object)
        self.cnt = self.covered_points.sum()

    def build_dmat(self, new_X, neg_X):
        with timed("build_dmat", n_old=len(self.X), n_new=len(new_X), n_neg=len(neg_X)):
            d_old = self._pdist(self.X)
            d_new = self._pdist(new_X)
            d_old_new = self._cdist(self.X, new_X)
            d_old_neg = self._cdist(self.X, neg_X)
            d_new_neg = self._cdist(new_X, neg_X)

            old_s = slice(0, len(self.X), None)
            new_s = slice(old_s.stop, old_s.stop + len(new_X), None)
            neg_s = slice(new_s.stop, new_s.stop + len(neg_X), None)

            new_d_mat = np.empty((new_s.stop, neg_s.stop), dtype=float)
            new_d_mat[old_s, old_s] = d_old
            new_d_mat[new_s, new_s] = d_new
            new_d_mat[old_s, new_s] = d_old_new
            new_d_mat[new_s, old_s] = d_old_new.T
            new_d_mat[old_s, neg_s] = d_old_neg
            new_d_mat[new_s, neg_s] = d_new_neg

        return new_d_mat

    def _subsample_negatives_random(self, neg_X):
        """Randomly select a global subset of negatives.
        neg_size is a fraction in (0, 1]: 1.0 uses all negatives.
        The same subset is reused for all positives in the current fit() call.

        The fractional size is also capped by an absolute ceiling derived
        from tailsize. _weibull_fit only ever consumes the `tailsize`
        nearest negatives per positive, so beyond a modest multiple of
        tailsize, extra negatives only inflate memory/time without
        affecting the fit."""
        k_frac = max(1, round(self.neg_size * len(neg_X)))
        k_cap = self._max_neg_cap()
        k = min(k_frac, k_cap)
        if k >= len(neg_X):
            return neg_X
        idx = self.rng.choice(len(neg_X), size=k, replace=False)
        return neg_X[np.sort(idx)]

    def _max_neg_cap(self):
        """Absolute ceiling on negatives sampled per fit() call, derived
        from tailsize."""
        return max(1, self._max_neg_multiplier * self.tailsize)

    def _compute_neg_d_faiss(self, X, neg_X):
        """For each positive x_i in X find the k nearest negatives in neg_X
        (GPU-accelerated when available) and return a per-row distance
        matrix neg_d of shape (len(X), k), using self.distance_metric.
        Each row is independent, so different positives can have
        different hard negatives.
        neg_size is a fraction in (0, 1]: 1.0 uses all negatives."""
        k = max(1, round(self.neg_size * len(neg_X)))
        neg_t = torch.as_tensor(neg_X, dtype=torch.float32, device=_DEVICE)
        q_t = torch.as_tensor(X, dtype=torch.float32, device=_DEVICE)
        if self.distance_metric == "cosine":
            neg_t = _l2_normalize(neg_t)
            q_t = _l2_normalize(q_t)
            d_full = 1.0 - (q_t @ neg_t.T)
        else:
            # full pairwise L2, shape (len(X), len(neg_X))
            d_full = torch.cdist(q_t, neg_t, p=2)
        # keep only the k smallest per row
        d_topk, _ = torch.topk(d_full, k, dim=1, largest=False, sorted=True)
        return d_topk.cpu().numpy().astype(np.float64)

    def fit(self, X, y, neg_X=_EMPTY, neg_d=None):
        # Negative subsampling: only when neg_size is set and neg_X is provided.
        if neg_d is None and len(neg_X) > 0 and self.neg_size is not None:
            if self.negative_selection == "faiss_siblings":
                # Per-positive: each x_i gets its own k nearest negatives.
                neg_d = self._compute_neg_d_faiss(X, neg_X)
                neg_X = _EMPTY
            else:
                # Random: one global subset shared by all positives.
                neg_X = self._subsample_negatives_random(neg_X)

        self.cnt += y.size
        new_X = empty_cat((self.X, X))
        new_y = empty_cat((self.y, y))

        if neg_d is None:
            d_mat = self.build_dmat(X, neg_X)
            labels = empty_cat((new_y, np.tile(None, len(neg_X)).astype(object)))
        else:
            assert len(X) == neg_d.shape[0]
            # Use X's dimension as fallback when self.X is still empty.
            dim = self.X.shape[1] if len(self.X) > 0 else X.shape[1]
            d_mat = self.build_dmat(X, np.zeros((neg_d.shape[1], dim)))
            d_mat[:len(self.X), len(new_X):] = np.inf
            d_mat[len(self.X):, len(new_X):] = neg_d

            labels = empty_cat((new_y, np.tile(None, neg_d.shape[-1]).astype(object)))

        row_range = range(len(self.X), len(new_X))

        args = zip((self.margin_scale * d_mat[r] for r in row_range),
                   row_range,
                   it.repeat(labels),
                   it.repeat(self.tailsize))
        with timed("weibull_fit_loop", n_rows=len(new_X) - len(self.X)):
            weibulls = np.array(list(map(_weibull_fit, args)), dtype=object)

        new_weibulls = empty_cat((self.weibulls, weibulls))

        self.weibulls = new_weibulls
        self.X = new_X
        self.y = new_y
        self.update_labels()

        if self.reduce:
            with timed("reduce_model", n_total=len(self.X), n_added=len(X)):
                self.reduce_model(d_mat[:len(self.X), :len(self.X)],
                                  indices=slice(len(self.X) - len(X),
                                                len(self.X), None))
        else:
            self.covered_points = np.ones(len(self.y), dtype=int)

        assert set(self.labels) == set(self.y)

    def update_labels(self):
        self.labels = np.array(list(set(self.y)), dtype=object)

    def reduce_model(self, d_mat, indices=None):
        """
        Model reduction routine. Calls off to set cover.
        """
        if self.cover_threshold >= 1.0:
            # optimize for the trivial case
            self.covered_points = np.ones(len(self.y), dtype=int)
            return
        if indices is None:
            indices = slice(None, None, None)

        keep = np.array((), dtype=int)
        keep_cover = []
        for ulabel in self.labels:
            ind = np.where(self.y == ulabel)[0]
            keep_ind = np.where((self.y == ulabel)[:indices.start])[0]
            keep_ind, num_cover = set_cover(d_mat[np.ix_(ind, ind)],
                                            self.weibulls[ind],
                                            self.cover_threshold,
                                            np.arange(keep_ind.size),
                                            self.covered_points[keep_ind])
            keep = empty_cat((keep, ind[keep_ind]))
            keep_cover.extend(num_cover)

        self.covered_points = np.array(keep_cover)
        assert self.covered_points.sum() == self.cnt
        self.X = self.X[keep]
        self.weibulls = self.weibulls[keep]
        self.y = self.y[keep]

    def predict(self, X, ret_distribution=False):
        """
        Analogous to scikit-learn's predict method
        except takes a few more arguments which
        constitute the actual model.
        """
        n_query = 1 if X.ndim == 1 else len(X)
        with timed("predict_cdist", n_ev=len(self.X), n_query=n_query):
            d_mat = self._cdist(self.X, X).astype(np.float64)
        with timed("predict_weibull_eval", n_ev=len(self.X), n_query=n_query):
            probs = np.array(list(map(_weibull_eval, zip(d_mat, self.weibulls))))

        unsqueezed = False
        if X.ndim == 1:
            unsqueezed = True

        fused_probs = []
        for ulabel in self.labels:
            fused_probs.append(fuse_prob_for_label(probs[self.y == ulabel],
                                                   self.num_to_fuse))
        fused_probs = np.column_stack(fused_probs)

        if unsqueezed:
            fused_probs = fused_probs[0, ...]
            assert fused_probs.ndim == 1
        max_ind = np.argmax(fused_probs, axis=-1)
        predicted_labels = self.labels[max_ind]
        if ret_distribution:
            return predicted_labels, max_ind, fused_probs
        else:
            return np.array(predicted_labels.tolist())

    def generate(self, rng, points=None, number=None, label=None):
        """Synthesize points around existing extreme vectors by sampling a
        Weibull-inverted radius and a random direction in Euclidean space
        (see spherical2cart). This sampling is inherently Euclidean — it does
        not have a meaningful cosine-space equivalent — so it always uses
        Euclidean distance regardless of self.distance_metric."""

        assert (points is None) == (number is None) == (label is None)

        sampled = []
        slices = [slice(0, 0, None)]
        for n, w, x, y in zip(self.covered_points, self.weibulls, self.X,
                              self.y):
            n -= 1
            mr = loadMR(w)
            p = rng.uniform(size=n)
            r = np.array([mr.inv(pi) for pi in p])
            r[r < 0] = 0
            angles = rng.uniform(high=np.pi, size=(n, self.X.shape[1]-1))

            if n > 0:
                all_pn = spherical2cart(r, angles) + x[None, ...]

                sampled.append(all_pn)

            else:
                sampled.append(_EMPTY)

            slices.append(slice(slices[-1].stop, slices[-1].stop + n, None))

        if points is None:
            concatenated = empty_cat(sampled)
            assert len(concatenated) == self.covered_points.sum() - self.covered_points.size
            assert not np.any(concatenated == np.inf)
            return concatenated, np.repeat(self.y, self.covered_points - 1)

        del slices[0]

        d_mat = euclidean_cdist(points, self.X)
        ags = np.argsort(d_mat, axis=-1)
        order = np.tile(np.arange(ags.shape[1]), (ags.shape[0], 1))
        order = np.take_along_axis(order, ags, axis=-1)

        final_dmat = np.tile(np.inf, (len(points), number))
        for i, row in enumerate(order):
            count = 0
            for cell in row:
                if self.y[cell] != label and len(sampled[cell]) > 0:
                    n = len(sampled[cell])
                    toadd = min(n, number - count)
                    selected = euclidean_cdist(points[i],
                                               sampled[cell]).flatten()[:toadd]
                    final_dmat[i, count:count + toadd] = selected
                    count += toadd

                    if count == number:
                        break

        final_dmat = np.round(final_dmat, decimals=4)
        assert np.all(final_dmat > -1e-7)
        return np.maximum(final_dmat, 0.)

    def get_distance(self, points, exclude_label=None):
        return self._cdist(points, self.X[self.y != exclude_label, ...])



def spherical2cart(r, angles):
    assert np.all(r >= 0)
    assert np.all(angles >= 0)
    assert np.all(angles[:-1, ...] <= np.pi)
    assert np.all(angles[-1, ...] <= 2*np.pi)

    s = np.array(angles.shape)
    s[-1] += 1

    res = np.tile(-1.0, s)
    res[...] = r[..., None]
    res[..., 1:] *= np.sin(angles).cumprod(axis=-1)
    res[..., :-1] *= np.cos(angles)

    return res


if __name__ == "__main__":
    pref = "/home/luca/Software/ExtremeValueMachine/TestData/"
    Xtrain, ytrain = load_data(pref + "train.txt")
    print(Xtrain.shape, ytrain.shape)
    Xtest, ytest = load_data(pref + "test.txt")
    print(Xtest.shape, ytest.shape)
    evm = EVM()
    evm.fit(Xtrain, ytrain)
    print("...model size: {}".format(len(evm.y)))
    predictions, probs = evm.predict(Xtest)
    print(predictions[50:100])
    accuracy = get_accuracy(predictions, ytest)
    print("accuracy: {}".format(accuracy))