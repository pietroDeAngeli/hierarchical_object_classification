from pathlib import Path
import json

from collections import OrderedDict

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

import recsiam.agent as ag
import recsiam.data as data
import recsiam.embeddings as emb
import recsiam.memory as mem
import recsiam.openworld as ow
import recsiam.utils as utils
#import recsiam.models as models

from functools import partial

import torch

_EXP_DICT = {
        "seed": None,
        "n_exp": 1,
        "setting": {"type": "tree",
                    "setting_args": {"prob": 1.0}},
        "dataset": {"split_seed": None,
                    "descriptor": None,
                    "dl_args": {},
                    "ds_args": {},
                    "pre_embedded": False,
                    "metadata": [],
                    "meta_args": [{}]},
        "agent": {
                "bootstrap": 2,
                "max_neigh_check": 1,
                "fn": {"add_seen_element": "separate"},
                "remove": {"name": "random",
                           "args": {},
                           "seed": 2},
                "name": "online",
                "ag_args": {},
                "obj_mem_args": {},
                 },
        "model": {
                "embedding": "dinov3",
                #"emb_train": False,
                #"pretrained": True,
                #"aggregator": "mean",
                #"ag_args": {},
                #"pre_embed": True
                }
}


# DATASETS

def load_dataset_descriptor(path):
    path = Path(path)

    with path.open("r") as ifile:
        return json.load(ifile)


def prep_dataset(params, base_key="dataset"):

    if isinstance(params[base_key]["descriptor"], (str, Path)):
        desc = load_dataset_descriptor(params[base_key]["descriptor"])
    else:
        desc = params[base_key]["descriptor"]

    fac = data.train_factory(desc,
                             params[base_key]["split_seed"],
                             dl_args=params[base_key]["dl_args"],
                             ds_args=params[base_key]["ds_args"],
                             setting=params["setting"])

    return fac

# MODELS


def is_dynamic(params):
    return False


def prep_model(params):

    def instance_model():

        if params["dataset"]["pre_embedded"]:
            # se usi embedding già salvati
            return torch.nn.Identity()

        # embedding = stringa tipo "dinov3"
        emb_name = params["model"]["embedding"]

        emb_model_factory = emb.get_embedding(emb_name)
        model = emb_model_factory(
            pretrained=params["model"].get("pretrained", True)
        )

        return model

    return instance_model


def get_optimizer(key):
    return getattr(torch.optim, key)


_AGENT_FACT = {"online": ag.online_agent_factory,
               "active": ag.active_agent_factory}


def get_agent_factory(key):
    return _AGENT_FACT[key]


def prep_obj_mem(params):
    kwargs = params["agent"]["obj_mem_args"]

    def instance_obj_mem():
        return mem.ObjectsMemory(**kwargs)

    return instance_obj_mem


def prep_agent(params):
    ag_f = get_agent_factory(params["agent"]["name"])


    m_f = prep_model(params)

    kwargs = params["agent"]["ag_args"]
    kwargs = kwargs if kwargs is not None else {}

    return ag_f(m_f, memory_factory=prep_obj_mem(params),
                bootstrap=params["agent"]["bootstrap"], **kwargs)


def instance_ow_exp(params):
    a_f = prep_agent(params)
    d_f = prep_dataset(params)
    s_f = ow.supervisor_factory

    return ow.OpenWorld(a_f, d_f, s_f, params["seed"])


def do_experiments(*args, **kwargs):
    res = ow.do_experiment(*args, **kwargs)

    return res


def run_ow_exp(params, workers, quiet=False, torch_threads=1):
    exp = instance_ow_exp(params)
    gen = tqdm(exp.gen_experiments(params["n_exp"]), total=params["n_exp"], smoothing=0, disable=quiet)
    if torch_threads != -1:
        torch.set_num_threads(torch_threads)

    pool = Parallel(n_jobs=workers, batch_size=1)
    keys = ["metadata", "meta_args"]
    args_exp = {k: params["dataset"][k] for k in keys}

    results = pool(delayed(do_experiments)(*args,
                                           **args_exp)
                   for args in gen)
    sess_res = [r[0] for r in results]
    eval_res = [r[1] for r in results]

    return (ow.stack_results(sess_res), ow.stack_results(eval_res))
