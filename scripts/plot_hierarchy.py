import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tabulate import tabulate
import lz4.frame
import networkx as nx
import pickle

import recsiam.utils as utils

plt.rcParams.update({'font.size': 15})

_TITLE = True
_COLORS = "brgcmyk"
_STYLES = [c+"-" for c in _COLORS]


def smart_load(path):
    if path.endswith(".lz4"):
        with lz4.frame.open(str(path), mode="rb") as f:
            loaded = pickle.load(f)
    else:
        with open(path, "rb") as f:
            loaded = pickle.load(f)

    return loaded[0]


def main(cmdline):

    data = np.array([smart_load(f) for f in cmdline.i_files])

    if cmdline.labels is None:
        labels = ["data_"+str(i) for i in range(1, len(data)+1)]
    else:
        labels = cmdline.labels.split(",")

    assert len(labels) == len(data)

    basenames = ["cost", "up", "down", "hf", "sup"]
    base_path = Path(cmdline.output_file)
    if base_path.is_dir():
        outputs = [base_path / b for b in basenames]
    else:
        outputs = [str(base_path) + b for b in basenames]

    tables = [
#           np.vectorize(compute_conf_mat, signature="(),(),()->(n,m,k)")(data[:, 0, 0], data[:, 1, 0], data[:, 1, 1])[:, -1, ...]
            ]

    yss = [
        np.vectorize(compute_cost, signature="()->(n)")(data[:, 0]),
        np.vectorize(compute_up, signature="()->(n)")(data[:, 0]),
        np.vectorize(compute_down, signature="()->(n)")(data[:, 0]),
        np.vectorize(compute_hf, signature="()->(n)")(data[:, 0]),
        np.vectorize(compute_sup_prob, signature="()->(n)")(data[:, 0])
#        np.vectorize(compute_total_supervision, signature="()->(n)")(data[:, 1, 0]),
#        np.vectorize(compute_gen_diff_perf, signature="(),(),()->(n)")(data[:, 0, 0], data[:, 1, 0], data[:, 1, 1])
            ]

    discard = [10] * 5  
    m_discard = [1] * 5
    ylims = [None] * 5
    ylabs = ["geodesic distance"] * 3 + ["hF score"] + ["supervision"]
    xlabs = ["iteration"] * 5
    wins = [10] * 5

    for t in tables:
        print(tabulate(list(zip(labels, t))))

    for ys, o, d, w, ylim, yl, xl in zip(yss, outputs, discard, wins, ylims, ylabs, xlabs):
        plot_generic(ys, None, labels, o, d, w, ylim=ylim, ylab=yl, xlab=xl)

    for y, d in zip(yss, m_discard):
        print(tabulate(list(zip(labels, y[:, d:].mean(axis=1)))))




def plot_generic(ys, x, labels, output_file, discard, win, **kwargs):

    if x is None:
        x = np.arange(len(ys[0]))
    plt.clf()
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.grid()
    if "ylim" in kwargs and kwargs["ylim"] is not None:
        ax.set_ylim(*kwargs["ylim"])
    if "ylab" in kwargs and kwargs["ylab"] is not None:
        ax.set_ylabel(kwargs["ylab"])
    if "xlab" in kwargs and kwargs["xlab"] is not None:
        ax.set_xlabel(kwargs["xlab"])
#    ax.set_xlim(-5)
#    if _TITLE:
#        ax.set_title("Test results")

    for y, l, s in zip(ys, labels, _STYLES):
        ax.plot(x[discard:], moving_avg(y, win)[discard:], s, label=l)
    ax.legend()
    fig.tight_layout()
    #plt.subplots_adjust(top=0.99, bottom=0.08)
    fig.savefig(str(output_file) + ".png")


def moving_avg(data, win):
    if win == 1:
        return data
    f = np.tile(1. / win, win)

    res = np.zeros(data.size)
    res[:win - 1] = data[:win].mean()

    res[win - 1:] = np.convolve(data, f, mode="valid")

    return res


def compute_cost(sess_d):
    return sess_d["cost"].mean(axis=0).sum(axis=1)

def compute_up(sess_d):
    return compute_ud(sess_d, 0)

def compute_down(sess_d):
    return compute_ud(sess_d, 1)

def compute_ud(sess_d, ind):
    return sess_d["cost"].mean(axis=0)[:, ind]


def compute_sup_prob(res_s):
    s = res_s["ask"]
    return s.mean(axis=0)

def compute_hp_hr(sess_d):
    res = []
    for i in range(sess_d["pred"].shape[0]):
        res.append([compute_single_hp_hr(sess_d, i, j)
                    for j in range(sess_d["pred"].shape[1])])

    return np.asarray(res)

def compute_single_hp_hr(sess_d, i, j):

    if len(sess_d["hyer"][i, j]) == 0:
        return 1., 1.

    t = nx.DiGraph(sess_d["hyer"][i, j])

    pred = tuple(sess_d["pred"][i, j])
    sup = tuple(sess_d["sup"][i, j])

#    p_set = set(utils.gen_parents(pred[0], t)) | {pred[0], bool(pred[1])}
#    s_set = set(utils.gen_parents(sup[0], t)) | {sup[0], bool(sup[1])}
#    p_set = set(utils.gen_parents(pred[0], t)) | {pred[0]}
#    s_set = set(utils.gen_parents(sup[0], t)) | {sup[0]}
    p_set = set(utils.gen_parents(pred[0], t)) | {pred}
    s_set = set(utils.gen_parents(sup[0], t)) | {sup}

    p_c = len(p_set)
    s_c = len(s_set)
    ps_c = len(p_set & s_set)

    assert p_c > 0
    assert s_c > 0
    return ps_c / p_c + 1e-7, ps_c/s_c + 1e-7


def compute_hf(sess_d):
    np.seterr(divide='raise')
    hp_hr = compute_hp_hr(sess_d)

    fh = 2 * hp_hr[:, :, 0] * hp_hr[:, :, 1] / (hp_hr[:, :, 0] + hp_hr[:, :, 1])

    return fh.mean(axis=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("i_files", type=str, nargs='+',
                        help="list of file sto load")
    parser.add_argument("-o", "--output-file", default='plot', type=str,
                        help="output file name")
    parser.add_argument("--discard-first", default=10, type=int,
                        help="discard first N objects")
    parser.add_argument("-l", "--labels", default=None, type=str,
                        help="labels for files")
    args = parser.parse_args()

    result = main(args)

