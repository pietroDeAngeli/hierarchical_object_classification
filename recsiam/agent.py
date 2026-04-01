from __future__ import division

import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx

from functools import partial

from ignite.engine import Engine, Events

from . import utils
from .utils import a_app, t_app
from .sampling import SeadableRandomSampler

import logging

from . import memory as mem


def online_agent_template(model_factory, seed, supervisor, memory_factory,
                          bootstrap,  **kwargs):
    o_mem = memory_factory()
    s_mem = mem.SupervisionMemory()
    model = model_factory()
    ag = Agent(seed, o_mem, s_mem, model, supervisor, bootstrap, **kwargs)

    return ag


def active_agent_template(model_factory, seed, supervisor, memory_factory,
                          sup_effort, bootstrap, **kwargs):
    o_mem = memory_factory()
    s_mem = mem.SupervisionMemory()
    model = model_factory()
    ag = ActiveAgent(sup_effort, seed, o_mem, s_mem, model, supervisor,
                     bootstrap, **kwargs)

    return ag


def online_agent_factory(model_factory, **kwargs):
    return partial(online_agent_template, model_factory, **kwargs)


def active_agent_factory(model_factory, **kwargs):
    assert "sup_effort" in kwargs
    return partial(active_agent_template, model_factory, **kwargs)


def _t(shape):
    return np.ones(shape, dtype=np.bool_)


_T = _t(1)


def _f(shape):
    return np.zeros(shape, dtype=np.bool_)


_F = _f(1)


class Agent(object):

    def __init__(self, seed, obj_mem, sup_mem, model,
                 supervisior, bootstrap,
                 ):
        self.seed = seed
        self.obj_mem = obj_mem
        self.sup_mem = sup_mem
        self.model = model
        self.supervisor = supervisior

        self.bootstrap = bootstrap
        self.cnt = 0

    def in_bootstrap(self):
        return self.cnt < self.bootstrap

    def forward(self, data):
        self.model.eval()

        with torch.no_grad():
            embed = self.model(data)

        return embed

    def process_next(self, data, s_id):
        self.cnt += 1

        # Keep embed 2D (1, D) so ObjectsMemory.add_element tiles correctly
        embed = self.forward(data).cpu().numpy()

        if len(self.obj_mem) == 0:
            self.obj_mem.add_element(s_id, embed, utils.get_root(self.obj_mem.T),
                                     False, True, None)
            return None, 1.0, None, None, False

        if len(self.sup_mem) > 0:
            pred, prob,  ask = self.predict(embed, embedded=True)
        else:
            pred, prob, ask = (utils.get_root(self.obj_mem.T), False), 1.0, True

        if self.in_bootstrap():
            ask = True

        sup_ret = self.supervisor.ask_supervision(s_id, pred, self)
        sup, cost = sup_ret[:2], sup_ret[2]
        if ask:
            if not utils.is_ancestor(self.obj_mem.T,  pred[0], sup[0]):
                self.sup_mem.add_entry(False, prob)
            elif not sup[1] and utils.is_leaf(self.obj_mem.T, sup[0]):
                if len(self.obj_mem.T.nodes) > 1:
                    par = next(iter(self.obj_mem.T.pred[sup[0]]))
                    cls = self.obj_mem.T.nodes[par]["cls"]

                    res = cls.predict(embed, True)
                    cl_ind = np.where(cls.labels == sup[0])[0]
                    prob = res[2][:, cl_ind].max()
                    self.sup_mem.add_entry(True, prob)

            self.obj_mem.add_element(s_id, embed, sup[0], sup[1], True, pred)
        else:
            self.obj_mem.add_element(s_id, embed, pred[0], pred[1], False, pred)

        return pred, prob, sup, cost, ask

    def predict(self, data, embedded=False):
        if embedded:
            embeds = data
        else:
            embeds = self.forward([data])[0].numpy()

        if len(self.sup_mem) == 0:
            return (self.obj_mem.get_root(), False), 1.0, True
        thr = self.linear_threshold
        pred = self.obj_mem.predict(embeds, thr)
        probs = np.array([pred[2][i][j]
                          for i, j in zip(range(pred[1].size), pred[1])])
        amax = np.argmax(probs)

        final_pred = pred[0][amax]
        final_prob = probs[amax]

        return (final_pred, False), final_prob,  True

    @property
    def linear_threshold(self):
        if len(self.sup_mem) > 0:
            thr = mem.compute_linear_threshold(self.sup_mem.labels, self.sup_mem.distances)
        else:
            return 1.
        return thr

class ActiveAgent(Agent):

    def __init__(self, sup_effort, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sup_effort = sup_effort

    @property
    def window_thresholds(self):
        if len(self.sup_mem) > 0:
            thr = mem.compute_subtract_entropy_thresholds(self.sup_mem.labels,
                                                          self.sup_mem.distances, 
                                                          self.sup_effort)[:2]
        else:
            thr = 0., 1.
        return thr

    def predict(self, data, embedded=False):
        if embedded:
            embeds = data
        else:
            embeds = self.forward([data])[0].numpy()

        if len(self.sup_mem) == 0:
            return (self.obj_mem.get_root(), False), 1.0, True

        thr_r, thr_a = self.window_thresholds
        thr_l = self.linear_threshold
        pred = self.obj_mem.predict(embeds, thr_l)
        probs = np.array([pred[2][i][j]
                          for i, j in zip(range(pred[1].size), pred[1])])
        amax = np.argmax(probs)

        final_pred = pred[0][amax]
        final_prob = probs[amax]

        get_sup = final_prob > thr_r and final_prob < thr_a

        return (final_pred, False), final_prob,  get_sup


def online_decide(distance, sup_mem):
    distance = np.asarray(distance)
    thr = mem.compute_linear_threshold(sup_mem.labels, sup_mem.distances)

    return thr > distance, _t(distance.shape)


def active_decide_by_entropy(distance, sup_mem, fraction):
    distance = np.asarray(distance)

    l_thr, u_thr, l_i, u_i, _ = mem.compute_subtract_entropy_thresholds(sup_mem.labels, sup_mem.distances, fraction)

    sup_s = slice(l_i, u_i + 1)

    thr = mem.compute_linear_threshold(sup_mem.labels[sup_s], sup_mem.distances[sup_s])

    return thr > distance, (distance > l_thr) & (distance < u_thr)


def active_decide_by_consensus(distance, sup_mem, fraction):
    distance = np.asarray(distance)

    thr = mem.compute_linear_threshold(sup_mem.labels, sup_mem.distances)

    c_p = (sup_mem.distances < thr) == sup_mem.labels.astype(np.bool_)

    l_thr, u_thr = mem.old_compute_window_threshold(c_p, sup_mem.distances, fraction)

    return thr > distance, (distance > l_thr) & (distance < u_thr)
