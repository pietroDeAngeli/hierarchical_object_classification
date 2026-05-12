from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Sequence

import lz4.frame
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from . import eval as rec_eval
from . import init_hierarchy as init_h


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

def _load_split_embeddings(samples: Sequence[dict]):
    """Load and mean-pool embeddings for a list of samples.

    Returns
    -------
    X : np.ndarray, shape (n_samples, emb_dim)
    y : np.ndarray of object labels, shape (n_samples,)
    """
    X_chunks = []
    y_list = []
    for sample in samples:
        emb = init_h._load_embedding(sample["path"])
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb[None, :]
        # mean-pool over frame/patch dimension
        emb = emb.mean(axis=0)
        X_chunks.append(emb)
        y_list.append(sample["target"])

    if len(X_chunks) == 0:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=object)

    X = np.stack(X_chunks, axis=0)
    y = np.array(y_list, dtype=object)
    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_logistic_head(
    train_samples: Sequence[dict],
    C: float = 1.0,
    max_iter: int = 1000,
    solver: str = "lbfgs",
    n_jobs: int = 1,
):
    logging.info("Loading embeddings for %d training samples...", len(train_samples))
    X_train, y_train = _load_split_embeddings(train_samples)
    n_classes = len(np.unique(y_train))
    logging.info(
        "Training LogisticRegression: X=%s, classes=%d, C=%.4f, solver=%s, max_iter=%d",
        X_train.shape,
        n_classes,
        C,
        solver,
        max_iter,
    )

    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    # verbose=1 makes the solver print per-iteration loss/convergence info to stdout
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver=solver,
        n_jobs=n_jobs,
        verbose=1,
    )

    import time
    t0 = time.time()
    clf.fit(X_train, y_enc)
    elapsed = time.time() - t0

    converged = clf.n_iter_[0] < max_iter
    logging.info(
        "Training complete: iterations=%s converged=%s elapsed=%.1fs",
        clf.n_iter_.tolist(),
        converged,
        elapsed,
    )
    if not converged:
        logging.warning(
            "Solver did not converge (max_iter=%d). Consider increasing --max-iter.", max_iter
        )
    return clf, le


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_logistic_head(clf: LogisticRegression, le: LabelEncoder, test_samples: Sequence[dict]):
    logging.info("Loading embeddings for %d test samples...", len(test_samples))
    X_test, y_test = _load_split_embeddings(test_samples)

    if X_test.shape[0] == 0:
        logging.warning("No test samples to evaluate")
        return rec_eval.compute_eval_metrics([], [], tree=None)

    y_pred_enc = clf.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)

    logging.info("Evaluation complete")
    return rec_eval.compute_eval_metrics(y_test.tolist(), y_pred.tolist(), tree=None)


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def _save_model(clf: LogisticRegression, le: LabelEncoder, path_like):
    out_path = Path(path_like)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"clf": clf, "le": le}
    with lz4.frame.open(str(out_path), mode="wb",
                        compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC) as f:
        pickle.dump(payload, f)
    logging.info("Model saved to %s", out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(cmdline):
    samples = init_h.build_samples_from_descriptor(cmdline.descriptor)
    logging.info("Loaded %d total samples from descriptor", len(samples))

    train_samples, test_samples = init_h.split_fixed_samples(
        samples,
        test_size=cmdline.test_size,
        train_size=cmdline.train_size,
        seed=cmdline.seed,
    )
    logging.info(
        "Split: %d train, %d test (seed=%d)",
        len(train_samples),
        len(test_samples),
        cmdline.seed,
    )

    if len(train_samples) == 0:
        raise ValueError("No train samples available after split")

    clf, le = train_logistic_head(
        train_samples,
        C=cmdline.C,
        max_iter=cmdline.max_iter,
        solver=cmdline.solver,
        n_jobs=cmdline.n_jobs,
    )

    summary = {
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "classes": int(len(le.classes_)),
        "C": float(cmdline.C),
        "max_iter": int(cmdline.max_iter),
        "solver": cmdline.solver,
        "seed": int(cmdline.seed),
    }

    if cmdline.eval_test:
        metrics = evaluate_logistic_head(clf, le, test_samples)
        summary["test_metrics"] = metrics
        logging.info("Test metrics: %s", metrics)

    if cmdline.output is not None:
        out_path = Path(cmdline.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as ofile:
            json.dump(summary, ofile, indent=1)
        logging.info("Summary saved to %s", out_path)

    if cmdline.model_output is not None:
        _save_model(clf, le, cmdline.model_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a logistic-regression classification head on pre-computed DINO embeddings."
    )
    parser.add_argument("--descriptor", required=True, type=str,
                        help="descriptor JSON path (must point to pre-embedded paths)")
    parser.add_argument("--output", default=None, type=str,
                        help="optional output JSON summary path")
    parser.add_argument("--model-output", default=None, type=str,
                        help="optional serialized model output (.lz4)")
    parser.add_argument("--test-size", default=0.15, type=float,
                        help="number of test samples (>=1) or fraction of total (0<v<1, e.g. 0.15 for 15%%)")
    parser.add_argument("--train-size", default=None, type=int,
                        help="fixed number of train samples after test split (default: all remaining)")
    parser.add_argument("--seed", default=0, type=int,
                        help="seed for deterministic train/test split")
    parser.add_argument("--C", default=1.0, type=float,
                        help="inverse regularization strength for LogisticRegression (default: 1.0)")
    parser.add_argument("--max-iter", default=1000, type=int,
                        help="max iterations for the solver (default: 1000)")
    parser.add_argument("--solver", default="lbfgs", type=str,
                        help="solver for LogisticRegression: lbfgs, saga, liblinear, ... (default: lbfgs)")
    parser.add_argument("--n-jobs", default=1, type=int,
                        help="number of parallel jobs (only effective with certain solvers/OvR)")
    parser.add_argument("--eval-test", action="store_true",
                        help="evaluate on test split and report metrics")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable debug logging")
    parser.add_argument("-q", "--quite", action="store_true",
                        help="disable warnings")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quite:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

    main(args)
