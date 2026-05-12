"""
Compute structural statistics of the pruned taxonomy hierarchy derived from a
descriptor JSON file.

Metrics reported
----------------
- num_nodes          : total number of nodes (leaves + internal)
- num_leaves         : nodes with no children (= dataset classes)
- num_internal       : nodes with at least one child
- depth_min/max/avg  : shortest/longest/average path from root to a leaf
- branch_min/max/avg : min/max/average out-degree of internal nodes only
"""

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import recsiam.utils as utils


def _find_root(T: nx.DiGraph):
    roots = [n for n in T.nodes if T.in_degree(n) == 0]
    if len(roots) != 1:
        raise ValueError("Tree has {} roots: {}".format(len(roots), roots))
    return roots[0]


def compute_hierarchy_stats(T: nx.DiGraph) -> dict:
    root = _find_root(T)

    leaves = [n for n in T.nodes if T.out_degree(n) == 0]
    internal = [n for n in T.nodes if T.out_degree(n) > 0]

    # Depth of each leaf (path length from root)
    leaf_depths = [nx.shortest_path_length(T, root, leaf) for leaf in leaves]

    # Branching factor of each internal node
    branch_factors = [T.out_degree(n) for n in internal]

    stats = {
        "num_nodes": len(T.nodes),
        "num_leaves": len(leaves),
        "num_internal": len(internal),
        "depth_min": int(np.min(leaf_depths)),
        "depth_max": int(np.max(leaf_depths)),
        "depth_avg": float(np.mean(leaf_depths)),
        "branch_min": int(np.min(branch_factors)) if branch_factors else None,
        "branch_max": int(np.max(branch_factors)) if branch_factors else None,
        "branch_avg": float(np.mean(branch_factors)) if branch_factors else None,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Print structural statistics of the pruned hierarchy in a descriptor."
    )
    parser.add_argument("descriptor", type=str,
                        help="Path to the descriptor JSON file")
    parser.add_argument("--flat", action="store_true",
                        help="Use a flat Root→class hierarchy instead of the WordNet hierarchy")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save stats as JSON")
    args = parser.parse_args()

    if args.flat:
        h = utils.flat_hierarchy_from_descriptor(args.descriptor)
    else:
        h = utils.hierarchy_from_descriptor(args.descriptor)

    T = utils.tree_from_list(h)
    stats = compute_hierarchy_stats(T)

    print("Hierarchy statistics for: {}{}".format(
        args.descriptor, " (flat)" if args.flat else " (pruned WordNet)"
    ))
    col = max(len(k) for k in stats) + 2
    for k, v in stats.items():
        if isinstance(v, float):
            print("  {:{}} {:.4f}".format(k, col, v))
        else:
            print("  {:{}} {}".format(k, col, v))

    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(stats, f, indent=2)
        print("Stats saved to", args.output)


if __name__ == "__main__":
    main()
