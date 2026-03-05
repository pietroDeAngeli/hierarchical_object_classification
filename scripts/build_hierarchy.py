import json
import os
import argparse
import shutil
from collections import defaultdict
from pathlib import Path
from graphviz import Digraph

# ====== Default Configuration ======
DEFAULT_VAL_JSON_PATH = "iNaturalist2021_val/val.json"
DEFAULT_OUTPUT_IMAGE_NAME = "taxonomy_hierarchy"
DEFAULT_KINGDOM_FILTER = "Animalia"
DEFAULT_MAX_CHILDREN = 2
DEFAULT_NUM_EXAMPLES = 10
DEFAULT_IMAGES_DIR = "iNaturalist2021_val"
DEFAULT_DATASET_DIR = "dataset"

LEVELS = ["kingdom", "phylum", "class", "order", "family", "genus", "name"]

def build_tree(categories, kingdom=DEFAULT_KINGDOM_FILTER):
    """Builds a nested dictionary tree filtered by kingdom."""
    tree = {}

    for cat in categories:
        if cat.get("kingdom", "").lower() != kingdom.lower():
            continue

        node = tree
        for level in LEVELS:
            val = cat.get(level)
            if not val:
                break
            if val not in node:
                node[val] = {}
            node = node[val]

    return tree

def prune_tree(tree, max_children=DEFAULT_MAX_CHILDREN):
    """Prunes the tree so that each node has at most max_children children."""
    pruned = {}
    for i, (key, subtree) in enumerate(tree.items()):
        if i >= max_children:
            break
        pruned[key] = prune_tree(subtree, max_children)
    return pruned

def add_nodes_edges(dot, tree, parent_id=None, level_idx=0):
    """Recursively adds nodes and edges to the Graphviz graph."""
    level_colors = {
        0: "lightyellow",   # Kingdom
        1: "lightcoral",    # Phylum
        2: "lightgreen",    # Class
        3: "lightskyblue",  # Order
        4: "plum",          # Family
        5: "peachpuff",     # Genus
        6: "lightgray",     # Species
    }
    color = level_colors.get(level_idx, "white")
    level_name = LEVELS[level_idx] if level_idx < len(LEVELS) else "?"

    for key, subtree in tree.items():
        node_id = f"{level_name}_{key}".replace(" ", "_").replace("/", "_")
        label = f"{level_name.capitalize()}\\n{key}"
        dot.node(node_id, label=label, fillcolor=color)
        if parent_id is not None:
            dot.edge(parent_id, node_id)
        add_nodes_edges(dot, subtree, parent_id=node_id, level_idx=level_idx + 1)

def sanitize_name(name):
    """Replace spaces with underscores for filesystem-safe names."""
    return name.replace(" ", "_")


def dict_to_nested_list(node):
    """Converts a nested dict tree to a nested list (descriptor.json format).
    Leaf nodes (empty dict) become string entries; internal nodes become lists.
    Leaf names are sanitized (spaces → underscores) to match folder names."""
    children = []
    for key, subtree in node.items():
        if not subtree:
            children.append(sanitize_name(key))  # leaf -> sanitized string
        else:
            children.append(dict_to_nested_list(subtree))  # internal -> recurse
    if len(children) == 1:
        return children[0]             # avoid single-element wrapping
    return children


def collect_leaves(node, leaves):
    """Collects all leaf keys (species names) in order."""
    for key, subtree in node.items():
        if not subtree:
            leaves.append(key)
        else:
            collect_leaves(subtree, leaves)


def build_image_index(data):
    """Builds a mapping from category name -> list of image file_names.
    Uses iNaturalist COCO-style fields: images, annotations, categories.
    """
    # category_id -> species name
    cat_id_to_name = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

    # image_id -> file_name
    img_id_to_file = {img["id"]: img["file_name"] for img in data.get("images", [])}

    # species name -> [file_names]
    name_to_files = defaultdict(list)
    for ann in data.get("annotations", []):
        cat_name = cat_id_to_name.get(ann["category_id"])
        file_name = img_id_to_file.get(ann["image_id"])
        if cat_name and file_name:
            name_to_files[cat_name].append(file_name)

    return name_to_files


def copy_examples(species_name, file_names, images_dir, dataset_dir, num_examples):
    """Copies up to num_examples images for a species into dataset_dir/<species_name>/.
    Returns the list of destination paths (relative to dataset_dir).
    """
    dest_dir = os.path.join(dataset_dir, species_name.replace(" ", "_"))
    os.makedirs(dest_dir, exist_ok=True)

    copied = []
    for fname in file_names[:num_examples]:
        src = os.path.join(images_dir, fname)
        if not os.path.exists(src):
            print(f"  [WARN] Image not found: {src}")
            continue
        dest = os.path.join(dest_dir, os.path.basename(fname))
        shutil.copy2(src, dest)
        copied.append(dest)

    return copied


def get_nested_list_depth(nested):
    """Returns the depth (number of edges from root to a leaf) of a nested list hierarchy."""
    if not isinstance(nested, list):
        return 0  # leaf string
    return 1 + max(get_nested_list_depth(child) for child in nested)


def update_configs_prob(inputs_dir, depth):
    """Updates the 'prob' array in all experiment JSON configs to match the tree depth.
    The depth passed here is the depth of the knowledge tree (excluding the instance level
    added by tree_with_instances), so prob must have exactly `depth` elements.
    Preserves the pattern: mass at first position or mass at last position.
    """
    inputs_dir = Path(inputs_dir)
    if not inputs_dir.exists():
        print(f"[WARN] inputs-dir '{inputs_dir}' not found, skipping config update.")
        return

    target_len = depth  # len(prob) == len(dag_longest_path) - 1 == depth

    for cfg_path in sorted(inputs_dir.glob("*.json")):
        with open(cfg_path) as f:
            cfg = json.load(f)

        setting = cfg.get("setting")
        if not setting or setting.get("type") != "tree":
            continue

        prob = setting.get("setting_args", {}).get("prob")
        if prob is None or len(prob) == target_len:
            continue

        # Preserve pattern: mass at first or last position
        if prob[0] == 1.0:
            new_prob = [1.0] + [0.0] * (target_len - 1)
        else:
            new_prob = [0.0] * (target_len - 1) + [1.0]

        cfg["setting"]["setting_args"]["prob"] = new_prob
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=1)

        print(f"  Updated '{cfg_path.name}': prob {prob} -> {new_prob}")


def save_metadata_json(tree, dataset_dir):
    """Saves {"hierarchy": [...], "depth": N} into dataset_dir/metadata.json.
    This file is automatically read by descriptor_from_filesystem() in fs2desc.py,
    which merges it with the scanned image paths to build the full descriptor.
    """
    hierarchy = dict_to_nested_list(tree)
    if not isinstance(hierarchy, list):
        hierarchy = [hierarchy]

    depth = get_nested_list_depth(hierarchy) + 1  # +1 for the instance level added by tree_with_instances

    os.makedirs(dataset_dir, exist_ok=True)
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({"hierarchy": hierarchy, "depth": depth}, f, indent=1)

    leaves = []
    collect_leaves(tree, leaves)
    print(f"metadata.json saved in '{dataset_dir}' ({len(leaves)} leaf classes, tree depth={depth})")
    return depth

def build_and_visualize_hierarchy(json_path, output_name, kingdom, max_children,
                                   images_dir, dataset_dir, num_examples, inputs_dir=None):
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} does not exist.")
        return

    print("Loading data...")
    with open(json_path, "r") as f:
        data = json.load(f)

    categories = data.get("categories", [])
    print(f"Total categories: {len(categories)}")

    print(f"Building tree for Kingdom='{kingdom}'...")
    tree = build_tree(categories, kingdom=kingdom)

    print(f"Pruning tree (max {max_children} children per node)...")
    pruned = prune_tree(tree, max_children=max_children)

    # Build image index and copy examples
    print(f"Building image index from annotations...")
    name_to_files = build_image_index(data)

    leaves = []
    collect_leaves(pruned, leaves)
    print(f"Copying up to {num_examples} images per species into '{dataset_dir}'...")
    for species in leaves:
        files = name_to_files.get(species, [])
        copied = copy_examples(species, files, images_dir, dataset_dir, num_examples)
        print(f"  {species}: {len(copied)} images copied")

    # Save metadata.json inside dataset_dir so fs2desc.py picks up the hierarchy
    depth = save_metadata_json(pruned, dataset_dir)

    # Update experiment configs prob arrays to match the actual tree depth
    if inputs_dir:
        print(f"Updating 'prob' arrays in '{inputs_dir}' configs to depth={depth}...")
        update_configs_prob(inputs_dir, depth)

    dot = Digraph(comment=f'Taxonomy: {kingdom}', format='pdf')
    dot.attr(rankdir='LR', splines='ortho')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='10')

    add_nodes_edges(dot, pruned)

    print(f"Rendering {output_name}.pdf...")
    try:
        dot.render(output_name, cleanup=True)
        print(f"Saved: {output_name}.pdf")
    except Exception as e:
        print(f"Error during rendering: {e}")
        print("Ensure Graphviz is installed:")
        print("  Linux: sudo apt install graphviz")
        print("  Python: pip install graphviz")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and visualize a taxonomic hierarchy from iNaturalist data.")
    parser.add_argument("--val-json", default=DEFAULT_VAL_JSON_PATH, help=f"Path to val.json (default: {DEFAULT_VAL_JSON_PATH})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_IMAGE_NAME, help=f"Output file name without extension (default: {DEFAULT_OUTPUT_IMAGE_NAME})")
    parser.add_argument("--kingdom", default=DEFAULT_KINGDOM_FILTER, help=f"Kingdom filter (default: {DEFAULT_KINGDOM_FILTER})")
    parser.add_argument("--max-children", type=int, default=DEFAULT_MAX_CHILDREN, help=f"Max children per node (default: {DEFAULT_MAX_CHILDREN})")
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR, help=f"Root directory containing iNaturalist images (default: {DEFAULT_IMAGES_DIR})")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, help=f"Destination directory for copied images (default: {DEFAULT_DATASET_DIR})")
    parser.add_argument("--num-examples", type=int, default=DEFAULT_NUM_EXAMPLES, help=f"Number of example images to copy per species (default: {DEFAULT_NUM_EXAMPLES})")
    parser.add_argument("--inputs-dir", default=None, help="Path to the inputs/ directory; if provided, updates 'prob' arrays in experiment configs to match tree depth")
    args = parser.parse_args()

    build_and_visualize_hierarchy(
        args.val_json, args.output, args.kingdom, args.max_children,
        args.images_dir, args.dataset_dir, args.num_examples, args.inputs_dir
    )
