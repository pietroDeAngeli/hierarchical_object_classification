import numpy as np
import libmr
import sklearn.metrics.pairwise
import itertools as it
import logging
from .utils import empty_cat


_EMPTY = np.array(())


def euclidean_cdist(X, Y):
    if len(X) > 0 and len(Y) > 0:
        if X.ndim == 1:
            X = X[None, :]
        if Y.ndim == 1:
            Y = Y[None, :]
        result = sklearn.metrics.pairwise.pairwise_distances(X, Y,
                                                           metric="euclidean",
                                                           n_jobs=1)
#        sq = np.squeeze(result)
#        if sq.ndim == 0:
#            sq = sq[None, ...]

        return result
    else:
        return np.zeros(shape=(len(X), len(Y)))


def euclidean_pdist(X):
    if len(X) > 0:
        if X.ndim == 1:
            X = X[None, :]
        return sklearn.metrics.pairwise.pairwise_distances(X,
                                                           metric="euclidean",
                                                           n_jobs=1)
    else:
        return _EMPTY


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
    assert not np.all(nearest == nearest[0])
    mr.fit_low(nearest, tailsize)

    ret = str(mr)
    assert ret != ''
    return mr.get_params()


def set_cover_greedy(universe, subsets, keep_ind, covered_points):
    """
    A greedy approximation to Set Cover.
    """
    universe = set(universe)
    subsets = list(map(set, subsets))
    covered = 0
    new_cover = set()
    len_a = np.array([len(s) for s in subsets])

    res = []
    num_cover = []
    k = 0

    already_covered = set(keep_ind)
    for s in subsets:
        s -= already_covered
    for i in keep_ind:
        subsets[i] |= {i}

    while covered < len(universe) or k < keep_ind.size:
        for i, s in enumerate(subsets):
            s -= new_cover
            len_a[i] = len(s)

        if k < keep_ind.size:
            max_index = keep_ind[k]
            previous_cover = covered_points[k] - 1
            k += 1
        else:
            max_index = len_a.argmax()
            previous_cover = 0

        new_cover = set(subsets[max_index])
        covered += len(new_cover)
        res.append(max_index)

        final_cover_value = len(new_cover) + previous_cover
        assert final_cover_value > 0
        num_cover.append(final_cover_value)

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
    if prob_mat.shape[0] == 0:
        return np.zeros(prob_mat.shape[1])
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
                 reduce=True):

        self.evt_indices = evt_indices
        self.margin_scale = margin_scale
        self.num_to_fuse = num_to_fuse
        self.tailsize = tailsize
        self.cover_threshold = cover_threshold
        self.reduce = bool(reduce)
        self.cnt = 0

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
                "reduce": self.reduce
               }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def copy(self):
        new = EVM(evt_indices=self.evt_indices,
                  margin_scale=self.margin_scale,
                  num_to_fuse=self.num_to_fuse,
                  tailsize=self.tailsize,
                  cover_threshold=self.cover_threshold,
                  reduce=self.reduce)

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
        d_old = euclidean_pdist(self.X)
        d_new = euclidean_pdist(new_X)
        d_old_new = euclidean_cdist(self.X, new_X)
        d_old_neg = euclidean_cdist(self.X, neg_X)
        d_new_neg = euclidean_cdist(new_X, neg_X)

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

    def fit(self, X, y, neg_X=_EMPTY, neg_d=None):

        self.cnt += y.size
        new_X = empty_cat((self.X, X))
        new_y = empty_cat((self.y, y))

        if neg_d is None:
            d_mat = self.build_dmat(X, neg_X)
            labels = empty_cat((new_y, np.tile(None, len(neg_X)).astype(object)))
        else:
            assert len(X) == neg_d.shape[0]
            d_mat = self.build_dmat(X, np.zeros((neg_d.shape[1],
                                                 self.X.shape[1])))
            d_mat[:len(self.X), len(new_X):] = np.inf
            d_mat[len(self.X):, len(new_X):] = neg_d

            labels = empty_cat((new_y, np.tile(None, neg_d.shape[-1]).astype(object)))

        row_range = range(len(self.X), len(new_X))

        args = zip((self.margin_scale * d_mat[r] for r in row_range),
                   row_range,
                   it.repeat(labels),
                   it.repeat(self.tailsize))
        weibulls = np.array(list(map(_weibull_fit, args)), dtype=object)

        new_weibulls = empty_cat((self.weibulls, weibulls))

        self.weibulls = new_weibulls
        self.X = new_X
        self.y = new_y
        self.update_labels()

        if self.reduce:
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
        d_mat = euclidean_cdist(self.X, X).astype(np.float64)
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
        return euclidean_cdist(points, self.X[self.y != exclude_label, ...])



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
