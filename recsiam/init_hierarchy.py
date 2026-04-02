from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import lz4.frame
import numpy as np

from . import memory as mem
from . import utils
from . import eval as rec_eval


DEFAULT_OBJ_MEM_ARGS = {
	"evm_args": {
		"cover_threshold": 0.5,
		"margin_scale": 1.0,
		"num_to_fuse": 1,
		"reduce": True,
		"tailsize": 25,
	},
	"update_policy": "recompute",
}

def memory_from_descriptor(descriptor, obj_mem_args, flat_hierarchy=False):
	if flat_hierarchy:
		hierarchy = utils.flat_hierarchy_from_descriptor(descriptor)
	else:
		desc = utils.load_descriptor(descriptor)
		hierarchy = utils.hierarchy_from_descriptor(desc)

	kwargs = dict(obj_mem_args)
	evm_args = kwargs.pop("evm_args")

	return mem.StaticHierarchyMemory(hierarchy=hierarchy, evm_args=evm_args, **kwargs)


def build_samples_from_descriptor(descriptor):
	desc = utils.load_descriptor(descriptor)
	if not isinstance(desc, (list, tuple)) or len(desc) < 2:
		raise ValueError("Descriptor must be a 2-element sequence: (info, objects)")

	objects = desc[1]
	samples = []
	for obj in objects:
		target = obj.get("name", obj.get("id"))
		for path in obj.get("paths", []):
			samples.append({"target": target, "path": path})

	if len(samples) == 0:
		raise ValueError("Descriptor does not contain any sample path")

	return samples


def split_fixed_samples(samples, test_size=0, train_size=None, seed=0):
	if test_size < 0:
		raise ValueError("test_size must be >= 0")
	if train_size is not None and train_size < 0:
		raise ValueError("train_size must be >= 0")

	total = len(samples)
	if test_size > total:
		raise ValueError("test_size ({}) is larger than available samples ({})".format(test_size, total))

	rng = np.random.RandomState(seed)
	perm = rng.permutation(total)

	test_idx = perm[:test_size]
	rem_idx = perm[test_size:]

	if train_size is None:
		train_idx = rem_idx
	else:
		if train_size > rem_idx.size:
			raise ValueError(
				"train_size ({}) is larger than remaining samples after test split ({})".format(
					train_size, rem_idx.size
				)
			)
		train_idx = rem_idx[:train_size]

	train_samples = [samples[i] for i in train_idx]
	test_samples = [samples[i] for i in test_idx]

	return train_samples, test_samples


def _load_embedding(path):
	path = Path(path)
	if path.suffix == ".lz4":
		with lz4.frame.open(str(path), mode="rb") as f:
			arr = np.load(f)
	else:
		arr = np.load(str(path), allow_pickle=True)

	arr = np.asarray(arr)
	if arr.ndim == 1:
		arr = arr[None, :]

	return arr


def fit_memory_from_samples(obj_mem, train_samples, log_every=100):
	total = len(train_samples)
	if total == 0:
		logging.info("Training skipped: no train samples available")
		return

	logging.info("Starting static-memory training on %d samples", total)
	for idx, sample in enumerate(train_samples):
		emb = _load_embedding(sample["path"])
		target = sample["target"]

		emb = np.asarray(emb)
		if emb.ndim == 1:
			emb = emb[None, :]

		obj_mem.add_element(new_id="sample_{}".format(idx),
							new_data=emb,
							target=target,
							new_genus=False,
							supervised=True,
							pred=None)

		if log_every is not None and log_every > 0:
			seen = idx + 1
			if (seen % int(log_every) == 0) or (seen == total):
				progress = 100.0 * float(seen) / float(total)
				logging.info(
					"Training progress: %d/%d (%.1f%%) pending_updates=%d",
					seen,
					total,
					progress,
					int(getattr(obj_mem, "_pending_update_count", len(getattr(obj_mem, "_pending_updates", [])))),
				)

	if hasattr(obj_mem, "finalize_updates"):
		obj_mem.finalize_updates()
	logging.info("Training completed: processed %d/%d samples", total, total)


def evaluate_test_samples(obj_mem, test_samples, thr_a=0.0, thr_r=None):
	return rec_eval.evaluate_test_samples(
		obj_mem,
		test_samples,
		embedding_loader=_load_embedding,
		thr_a=thr_a,
		thr_r=thr_r,
		logger=logging.getLogger("recsiam.init_hierarchy.eval"),
	)


def train_static_batch(obj_mem, embeddings: Sequence[np.ndarray], targets: Sequence,
					   supervised: bool = True):
	if len(embeddings) != len(targets):
		raise ValueError("embeddings and targets must have same length")

	for idx, (emb, target) in enumerate(zip(embeddings, targets)):
		emb = np.asarray(emb)
		if emb.ndim == 1:
			emb = emb[None, :]
		obj_mem.add_element(new_id="sample_{}".format(idx),
							new_data=emb,
							target=target,
							new_genus=False,
							supervised=supervised,
							pred=None)


def _summary(obj_mem):
	leaves = sum(1 for n in obj_mem.T.nodes if utils.is_leaf(obj_mem.T, n))
	internals = len(obj_mem.T.nodes) - leaves
	return {
		"nodes": len(obj_mem.T.nodes),
		"leaf_nodes": leaves,
		"internal_nodes": internals,
		"seen_samples": int(getattr(obj_mem, "_seen", 0)),
		"evm_batch_size": int(getattr(obj_mem, "evm_batch_size", 1)),
		"pending_updates": int(getattr(obj_mem, "_pending_update_count", len(getattr(obj_mem, "_pending_updates", [])))),
		"pending_leaf_updates": int(len(getattr(obj_mem, "_pending_leaf_updates", []))),
	}


def _extract_obj_mem_args(payload):
	if not isinstance(payload, dict):
		raise ValueError("Object memory args payload must be a JSON object")

	if "evm_args" in payload:
		return payload

	if "obj_mem_args" in payload and isinstance(payload["obj_mem_args"], dict):
		return payload["obj_mem_args"]

	agent_cfg = payload.get("agent")
	if isinstance(agent_cfg, dict) and isinstance(agent_cfg.get("obj_mem_args"), dict):
		return agent_cfg["obj_mem_args"]

	raise ValueError(
		"Could not extract object memory args. Expected one of: "
		"{evm_args: ...}, {obj_mem_args: ...}, or {agent: {obj_mem_args: ...}}"
	)


def _load_obj_mem_args(path_like, descriptor_path=None):
	args_path = Path(path_like)
	candidates = [args_path]

	if descriptor_path is not None:
		desc_dir = Path(descriptor_path).resolve().parent
		if not args_path.is_absolute():
			candidates.append(desc_dir / args_path)

	for candidate in candidates:
		if candidate.exists():
			with candidate.open("r") as ifile:
				payload = json.load(ifile)
			return _extract_obj_mem_args(payload)

	logging.warning(
		"Object memory args file '%s' not found (tried: %s). Falling back to defaults.",
		str(path_like),
		", ".join(str(p) for p in candidates),
	)
	return dict(DEFAULT_OBJ_MEM_ARGS)


def main(cmdline):
	obj_mem_args = _load_obj_mem_args(cmdline.obj_mem_args,
								  descriptor_path=cmdline.descriptor)

	if cmdline.evm_batch_size is not None:
		obj_mem_args["evm_batch_size"] = cmdline.evm_batch_size

	samples = build_samples_from_descriptor(cmdline.descriptor)
	train_samples, test_samples = split_fixed_samples(
		samples,
		test_size=cmdline.test_size,
		train_size=cmdline.train_size,
		seed=cmdline.seed,
	)

	obj_mem = memory_from_descriptor(cmdline.descriptor,
								 obj_mem_args,
								 flat_hierarchy=cmdline.flat_hierarchy)

	if cmdline.fit_train:
		fit_memory_from_samples(obj_mem,
						 train_samples,
						 log_every=cmdline.train_log_every)
	elif cmdline.eval_test:
		logging.warning("--eval-test requested without --fit-train: metrics reflect an unfitted memory")

	info = _summary(obj_mem)
	info["available_samples"] = len(samples)
	info["train_samples"] = len(train_samples)
	info["test_samples"] = len(test_samples)
	info["flat_hierarchy"] = bool(cmdline.flat_hierarchy)

	if cmdline.eval_test:
		metrics = evaluate_test_samples(obj_mem,
							   test_samples,
							   thr_a=cmdline.thr_a,
							   thr_r=cmdline.thr_r)
		info["test_metrics"] = metrics
		logging.info("Test evaluation metrics: %s", metrics)

	logging.info("Initialized static hierarchy memory: %s", info)

	if cmdline.test_output is not None:
		test_payload = {
			"seed": cmdline.seed,
			"test_size": cmdline.test_size,
			"available_samples": len(samples),
			"samples": test_samples,
			"eval_test": bool(cmdline.eval_test),
		}
		if cmdline.eval_test:
			test_payload["metrics"] = info.get("test_metrics")
		test_output_path = Path(cmdline.test_output)
		test_output_path.parent.mkdir(parents=True, exist_ok=True)
		with test_output_path.open("w") as ofile:
			json.dump(test_payload, ofile, indent=1)

	if cmdline.output is not None:
		output_path = Path(cmdline.output)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		with output_path.open("w") as ofile:
			json.dump(info, ofile, indent=1)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--descriptor", required=True, type=str,
						help="descriptor JSON path")
	parser.add_argument("--obj-mem-args", required=True, type=str,
						help="JSON file with object memory args (must include evm_args)")
	parser.add_argument("--output", default=None, type=str,
						help="optional output path for a small initialization summary")
	parser.add_argument("--test-size", default=0, type=int,
						help="fixed number of examples to reserve for test set")
	parser.add_argument("--train-size", default=None, type=int,
						help="fixed number of train examples after test split (default: all remaining)")
	parser.add_argument("--seed", default=0, type=int,
						help="random seed used for reproducible train/test sampling")
	parser.add_argument("--fit-train", action="store_true",
						help="fit static memory on sampled train examples")
	parser.add_argument("--eval-test", action="store_true",
						help="evaluate test split and report metrics")
	parser.add_argument("--thr-a", default=0.0, type=float,
						help="acceptance threshold used during prediction for test eval")
	parser.add_argument("--thr-r", default=None, type=float,
						help="rejection threshold used during prediction for test eval (default: thr-a)")
	parser.add_argument("--test-output", default=None, type=str,
						help="optional output path storing sampled test set metadata")
	parser.add_argument("--evm-batch-size", default=None, type=int,
						help="update EVMs every N inserted samples (default: 1)")
	parser.add_argument("--flat-hierarchy", action="store_true",
						help="build a flat hierarchy: Root with class nodes as direct children")
	parser.add_argument("--train-log-every", default=100, type=int,
						help="log training progress every N processed samples (<=0 disables periodic logs)")
	parser.add_argument("-v", "--verbose", action="store_true",
						help="triggers verbose mode")
	parser.add_argument("-q", "--quite", action="store_true",
						help="do not output warnings")
	args = parser.parse_args()

	if args.verbose:
		logging.basicConfig(level=logging.DEBUG)
	elif args.quite:
		logging.basicConfig(level=logging.ERROR)
	else:
		logging.basicConfig(level=logging.INFO)

	main(args)
