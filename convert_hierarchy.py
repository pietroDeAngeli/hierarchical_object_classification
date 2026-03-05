"""
Converts taxonomy_hierarchy.json (nested dict format) to descriptor.json format.

Input format (taxonomy_hierarchy.json):
    {"Kingdom": {"Phylum": {"Class": {..., "Species": {}}}}}

Output format (descriptor.json):
    [
        {"hierarchy": [...nested list of class names...]},
        [{"id": 0, "name": "Species name", "paths": []}, ...]
    ]
"""
import json
from pathlib import Path


def dict_to_nested_list(node: dict):
    """
    Recursively converts a nested dict to a nested list.
    - Internal nodes (dict with children): becomes a list of converted children
    - Leaf nodes (empty dict): returns the key (handled by the caller)
    """
    # If the node has no children, it's handled by the parent as a leaf
    children = []
    for key, subtree in node.items():
        if not subtree:           # empty dict -> leaf = class name string
            children.append(key)
        else:                     # non-empty dict -> recurse
            children.append(dict_to_nested_list(subtree))
    # If only one child, flatten (avoids single-element wrapping lists)
    if len(children) == 1:
        return children[0]
    return children


def collect_leaves(node: dict, leaves: list):
    """Collect all leaf keys (species names) in order."""
    for key, subtree in node.items():
        if not subtree:
            leaves.append(key)
        else:
            collect_leaves(subtree, leaves)


def convert(input_path: str, output_path: str, paths_root: str = ""):
    with open(input_path) as f:
        taxonomy = json.load(f)

    # Build nested list hierarchy
    # taxonomy is {"Kingdom": {...}} -> wrap the top-level in a list
    # to make the root a virtual internal node (matches descriptor.json structure)
    hierarchy = dict_to_nested_list(taxonomy)

    # If the root is not a list (single kingdom), wrap it
    if not isinstance(hierarchy, list):
        hierarchy = [hierarchy]

    # Collect all leaf species names in order
    leaves = []
    collect_leaves(taxonomy, leaves)

    # Build objects list
    objects = []
    for idx, name in enumerate(leaves):
        obj = {"id": idx, "name": name, "paths": []}
        if paths_root:
            # Optionally populate paths from filesystem
            class_dir = Path(paths_root) / name.replace(" ", "_")
            if class_dir.exists():
                obj["paths"] = sorted(str(p) for p in class_dir.iterdir() if p.is_file())
        objects.append(obj)

    descriptor = [{"hierarchy": hierarchy}, objects]

    with open(output_path, "w") as f:
        json.dump(descriptor, f, indent=1)

    print(f"Converted {len(leaves)} classes.")
    print(f"Hierarchy depth (longest path): {get_depth(taxonomy)} levels")
    print(f"Output saved to: {output_path}")


def get_depth(node: dict, current=0) -> int:
    if not node:
        return current
    return max(get_depth(v, current + 1) for v in node.values())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert taxonomy_hierarchy.json to descriptor.json")
    parser.add_argument("--input",  default="taxonomy_hierarchy.json",
                        help="Path to taxonomy_hierarchy.json")
    parser.add_argument("--output", default="descriptor_taxonomy.json",
                        help="Path to output descriptor JSON")
    parser.add_argument("--paths-root", default="",
                        help="Optional: root folder containing per-class subfolders with images")
    args = parser.parse_args()

    convert(args.input, args.output, args.paths_root)
