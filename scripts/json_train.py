import argparse
import logging
import json
import tempfile
import pickle
import copy
import torch
import numpy as np
from pathlib import Path
import recsiam.cfghelpers as cfg
import recsiam.utils as utils
import re
import lz4.frame

np.seterr(all='raise')

def set_torch_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_descriptor_fixed(descriptor, test_size, seed):
    desc = utils.load_descriptor(descriptor)
    if not isinstance(desc, (list, tuple)) or len(desc) < 2:
        raise ValueError("Descriptor must be a 2-element sequence: (info, objects)")

    info = copy.deepcopy(desc[0])
    objects = copy.deepcopy(desc[1])

    entries = []
    for obj_idx, obj in enumerate(objects):
        target = obj.get("name", obj.get("id"))
        for path in obj.get("paths", []):
            entries.append({"obj_idx": obj_idx, "target": target, "path": path})

    total = len(entries)
    if test_size < 0:
        raise ValueError("test_size must be >= 0")
    if test_size > total:
        raise ValueError("test_size ({}) is larger than available samples ({})".format(test_size, total))

    rng = np.random.RandomState(seed)
    perm = rng.permutation(total)
    test_sel = set(perm[:test_size].tolist())

    train_objects = []
    test_objects = []
    test_samples = []
    curr = 0
    for obj in objects:
        paths = obj.get("paths", [])
        keep_paths = []
        test_paths = []
        for path in paths:
            if curr in test_sel:
                test_samples.append({"target": obj.get("name", obj.get("id")), "path": path})
                test_paths.append(path)
            else:
                keep_paths.append(path)
            curr += 1

        new_obj = dict(obj)
        new_obj["paths"] = keep_paths
        if len(keep_paths) > 0:
            train_objects.append(new_obj)

        test_obj = dict(obj)
        test_obj["paths"] = test_paths
        if len(test_paths) > 0:
            test_objects.append(test_obj)

    return (info, train_objects), (copy.deepcopy(info), test_objects), test_samples, total


def main(cmdline):

    params = json.loads(Path(cmdline.json).read_text())

    params_run = copy.deepcopy(params)

    if cmdline.test_size > 0:
        desc = params_run["dataset"]["descriptor"]
        split_seed = cmdline.test_seed
        if split_seed is None:
            split_seed = params_run["dataset"].get("split_seed", 0)

        train_desc, test_desc, test_samples, total_samples = _split_descriptor_fixed(
            desc,
            test_size=cmdline.test_size,
            seed=split_seed,
        )

        params_run["dataset"]["descriptor"] = train_desc
        params_run["dataset"]["eval_descriptor"] = test_desc
        params["fixed_test_split"] = {
            "enabled": True,
            "seed": int(split_seed),
            "test_size": int(cmdline.test_size),
            "available_samples": int(total_samples),
            "train_samples": int(total_samples - cmdline.test_size),
        }

        if cmdline.test_output is not None:
            out_path = Path(cmdline.test_output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "seed": int(split_seed),
                "test_size": int(cmdline.test_size),
                "available_samples": int(total_samples),
                "samples": test_samples,
            }
            with out_path.open("w") as ofile:
                json.dump(payload, ofile, indent=1)

        logging.info(
            "Using fixed test split: test=%d train=%d total=%d seed=%d",
            cmdline.test_size,
            total_samples - cmdline.test_size,
            total_samples,
            split_seed,
        )
    else:
        params["fixed_test_split"] = {"enabled": False}

    results = cfg.run_ow_exp(params_run, cmdline.workers, torch_threads=cmdline.threads)

    if cmdline.eval_test and cmdline.test_size > 0:
        eval_res = results[1]
        # eval_res["metrics"] is an array of dicts, one per experiment run
        metrics_list = eval_res["metrics"].tolist()
        agg_metrics = {}
        for key in ["accuracy", "precision", "f1", "mean_geodesic_distance"]:
            vals = [m[key] for m in metrics_list if isinstance(m, dict) and m.get(key) is not None]
            agg_metrics[key] = float(np.mean(vals)) if vals else None
        for key in ["total_nodes", "leaf_nodes"]:
            vals = [v for v in eval_res[key].tolist() if v is not None]
            agg_metrics[key] = float(np.mean(vals)) if vals else None
        logging.info("Aggregated test metrics over %d runs: %s", len(metrics_list), agg_metrics)

        if cmdline.test_output is not None:
            # re-open and update the test_output JSON written above
            test_output_path = Path(cmdline.test_output)
            if test_output_path.exists():
                with test_output_path.open("r") as ifile:
                    test_payload = json.load(ifile)
            else:
                test_payload = {}
            test_payload["eval_test"] = True
            test_payload["metrics"] = agg_metrics
            with test_output_path.open("w") as ofile:
                json.dump(test_payload, ofile, indent=1)

    if cmdline.results is None:
        outfile, outfile_path = tempfile.mkstemp(prefix="json-train",
                                                 suffix=".npy.lz4")
        logging.info("storing results in {}".format(outfile_path))
    else:
        outfile_path = re.sub(r"\.npy$", "", re.sub(r"\.lz4$", "",
                                                    cmdline.results))
        outfile_path += ".npy.lz4"

    with lz4.frame.open(outfile_path,
                        mode="wb",
                        compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC) as f:
        pickle.dump((results, params), f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("json", type=str,
                        help="path containing the json to use")
    parser.add_argument("--results", type=str, default=None,
                        help="output file")
    parser.add_argument("--test-size", type=int, default=0,
                        help="fixed number of descriptor samples to reserve for test (excluded from training)")
    parser.add_argument("--test-seed", type=int, default=None,
                        help="seed for fixed test split (default: dataset split_seed from input json)")
    parser.add_argument("--test-output", type=str, default=None,
                        help="optional json path to store sampled test set metadata")
    parser.add_argument("--eval-test", action="store_true",
                        help="aggregate eval metrics from all runs and store in test-output JSON")
    parser.add_argument("-w", "--workers", type=int, default=-1,
                        help="number of joblib workers")
    parser.add_argument("-t", "--threads", type=int, default=1,
                        help="number of pytorch threads")

#       verbosity
    parser.add_argument("-v", "--verbose", action='store_true',
                        help="triggers verbose mode")
    parser.add_argument("-q", "--quite", action='store_true',
                        help="do not output warnings")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quite:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.results is not None:
        assert Path(args.results).parent.exists()
    main(args)
