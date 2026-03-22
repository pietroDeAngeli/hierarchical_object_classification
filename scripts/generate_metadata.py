import json
from collections import defaultdict
import os

def build_hierarchy(tree_file, output_file):
    nodes = []
    parents = []
    
    if not os.path.exists(tree_file):
        print(f"File {tree_file} not found!")
        return

    with open(tree_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                node = parts[0]
                parent_idx = int(parts[1])
                nodes.append(node)
                parents.append(parent_idx)

    children_map = defaultdict(list)
    roots = []
    for idx, (node, parent_idx) in enumerate(zip(nodes, parents)):
        if parent_idx == -1:
            roots.append(idx)
        else:
            children_map[parent_idx].append(idx)

    def get_subtree(idx):
        if idx not in children_map or len(children_map[idx]) == 0:
            return nodes[idx]
        return [nodes[idx], [get_subtree(child_idx) for child_idx in children_map[idx]]]

    subtrees = [get_subtree(r) for r in roots]
    
    # Aggiunge "Root" solo se ci sono alberi multipli indipendenti
    if len(subtrees) > 1:
        hierarchy = [["Root", subtrees]]
    else:
        hierarchy = subtrees

    output_data = {"hierarchy": hierarchy}
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Hierarchy successfully generated and saved to {output_file}")

if __name__ == "__main__":
    build_hierarchy('9k.tree', 'metadata.json')