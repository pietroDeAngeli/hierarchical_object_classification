from __future__ import division

import logging
import itertools
import numpy as np
import torch
from .supervision import supervisor_factory, get_supervision_stat
from . import utils


_OBJ_ID = "obj_id"
_SEQ_ID  = "s_id"
_NOSUP = (-123456789, False)
_NOPRED = _NOSUP
_NOCOST = (0, 0)


class OpenWorld(object):

    def __init__(self, agent_factory, dataset_factory,
                 supervisor_fac, seed):
        self.agent_factory = agent_factory
        self.dataset_factory = dataset_factory
        self.supervisor_factory = supervisor_fac
        self.seed = seed

        self.rnd = np.random.RandomState(self.seed)
        self.exp_seed, self.env_seed = self.rnd.randint(2**32, size=2)
        self.amb_seed = self.rnd.randint(2**32)

    def gen_experiments(self, n_exp):

        exp_seeds = get_exp_seeds(self.exp_seed, n_exp)
        env_seeds = get_exp_seeds(self.env_seed, n_exp)

        for exp_s, env_s in zip(exp_seeds, env_seeds):

            session_ds, eval_ds, inc_eval_ds = self.dataset_factory(env_s)
            supervisor = self.supervisor_factory(session_ds.dataset)
            agent = self.agent_factory(exp_s, supervisor)

            s_range = range(len(session_ds))

            yield agent, (session_ds, s_range), eval_ds


def get_exp_seeds(seed, n_exp):
    n_exp = np.asarray(n_exp)

    if n_exp.shape == ():
        n_exp = np.array([0, n_exp])

    elif n_exp.shape == (2,):
        pass

    else:
        raise ValueError("shape of n_exp is {}".format(n_exp.shape))

    rnd = np.random.RandomState(seed)
    seeds = rnd.randint(2**32, size=n_exp[1])[n_exp[0]:]

    return seeds


def counter(start=0):
    while True:
        yield start
        start += 1


def do_experiment(agent, session_seqs, eval_seqs, metadata=[], meta_args=[{}],
                  batch_size=1):
    logger = logging.getLogger("recsiam.openworld.do_experiment")

    logger.debug("session = {}\tsession len = {}\tsession range {}".format(
        session_seqs[0], len(session_seqs[0]), session_seqs[1]))

    session_pred = []
    session_id = []
    session_class = []
    session_ask = []
    session_sup = []
    session_metadata = [[] for _ in metadata]
    session_cost = []
    session_thr = []
    session_hyer = []
    session_prob = []

    eval_pred = []
    eval_class = []

    eval_metadata = [[] for _ in metadata]

    ev_data = []
    ev_data_len = []

    do_eval = False
    if do_eval:
        logger.debug("started evaluation data preload")
        for ds_ind, (data, obj_id) in enumerate(eval_seqs):
            eval_class.append(obj_id)

            ev_data.append(data[0])
            ev_data_len.append(len(data[0]))
            for i, (m, m_a) in enumerate(zip(metadata, meta_args)):
                meta = eval_seqs.dataset.get_metadata(m, ds_ind, **m_a)
                eval_metadata[i].append(meta)
                logger.debug("obj_id={}\tmet_keys = {}\tmeta_values = {}".format(obj_id, m, meta))

    logger.debug("started session")
    if batch_size > 1:
        # --- Batch training mode ---
        session_items = list(zip(*session_seqs))   # [(data, obj_id), s_id]
        for start in range(0, len(session_items), batch_size):
            chunk = session_items[start:start + batch_size]

            for i, (m, m_a) in enumerate(zip(metadata, meta_args)):
                for local_i, ((data, obj_id), s_id) in enumerate(chunk):
                    ds_ind = start + local_i
                    meta = session_seqs[0].dataset.get_metadata(m, ds_ind, **m_a)
                    session_metadata[i].append(meta)

            batch_input = [(data, s_id) for (data, obj_id), s_id in chunk]
            batch_results = agent.process_batch(batch_input)

            for (result, ((data, obj_id), s_id)) in zip(batch_results, chunk):
                pred, prob, sup, cost, ask = result

                session_pred.append(pred if pred is not None else _NOPRED)
                session_prob.append(prob)
                session_ask.append(ask)
                session_sup.append(sup if sup is not None else _NOSUP)
                session_id.append(s_id)
                session_class.append(obj_id)
                session_cost.append(cost if cost is not None else _NOCOST)

                if "window_thresholds" in agent.__class__.__dict__:
                    thr = agent.window_thresholds
                else:
                    thr = [agent.linear_threshold] * 2
                session_thr.append(thr)
                session_hyer.append(list(agent.obj_mem.T.edges))
    else:
        # --- Online (one sample at a time) mode ---
        for ds_ind, ((data, obj_id), s_id) in enumerate(zip(*session_seqs)):
            logger.debug("processing object {} ({} / {})".format(obj_id, ds_ind,
                                                                 len(session_seqs[1])))

            for i, (m, m_a) in enumerate(zip(metadata, meta_args)):
                meta = session_seqs[0].dataset.get_metadata(m, ds_ind, **m_a)
                session_metadata[i].append(meta)
                logger.debug("ds_ind ={}\tobj_id={}\tmet_keys = {}\tmeta_values = {}".format(ds_ind, obj_id, m, meta))

            # process next video
            pred, prob, sup, cost, ask = agent.process_next(data, s_id)

            session_pred.append(pred if pred is not None else _NOPRED)
            session_prob.append(prob)
            session_ask.append(ask)
            session_sup.append(sup if sup is not None else _NOSUP)
            session_id.append(s_id)
            session_class.append(obj_id)
            session_cost.append(cost if cost is not None else _NOCOST)

            if "window_thresholds" in agent.__class__.__dict__:
                thr = agent.window_thresholds
            else:
                thr = [agent.linear_threshold] * 2

            session_thr.append(thr)
            session_hyer.append(list(agent.obj_mem.T.edges))

        # validate / test
        if do_eval:
            e = np.array([agent.predict(elem)[0]
                          for elem in ev_data]).astype(int)
            eval_pred.append(e)

    s_d = {"pred": np.squeeze(session_pred),
           _SEQ_ID: np.squeeze(session_id),
           "ask": np.squeeze(session_ask),
           "prob": np.squeeze(session_prob),
           "sup": np.squeeze(session_sup),
           "cost": np.squeeze(session_cost),
           _OBJ_ID: np.squeeze(session_class),
           "thr": np.array(session_thr),
           "hyer": session_hyer,
           **{m: np.asarray(v) for m, v in zip(metadata, session_metadata)}
           }

    e_d = {"pred": np.squeeze(eval_pred),
           _OBJ_ID: np.squeeze(eval_class),
           **{m: np.asarray(v) for m, v in zip(metadata, eval_metadata)}
           }

    return s_d, e_d


def stack_results(res_l):

    stacked = {}
    for key in res_l[0]:
        if len(res_l) > 1:
            try:
                stacked[key] = np.array([r[key] for r in res_l])
            except ValueError:
                stacked[key] = np.array([r[key] for r in res_l], dtype=object)
        else:
            stacked[key] = res_l[0][key][None, ...]

    return stacked


def pad_to_dense(M):

    maxlen = max(len(r) for r in M)

    Z = np.zeros((len(M), maxlen))
    for enu, row in enumerate(M):
        Z[enu, :len(row)] += row
    return Z
