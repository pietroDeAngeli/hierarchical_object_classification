import argparse
import json

def process_dataset(path_in, path_out, labels_file="9k.labels", map_file="coco9k.map", names_file="coco.names"):
    # Load 9k.labels
    with open(labels_file, 'r') as f:
        wn_labels = [line.strip() for line in f]

    # Read coco.names and coco9k.map
    with open(names_file, 'r') as f:
        coco_names = [line.strip() for line in f]
    with open(map_file, 'r') as f:
        coco_idxs = [int(line.strip()) for line in f]

    if len(coco_names) != len(coco_idxs):
        raise ValueError(
            f"Mismatch between COCO names ({len(coco_names)}) and map entries ({len(coco_idxs)})."
        )

    invalid_map_indices = sorted({idx for idx in coco_idxs if idx < 0 or idx >= len(wn_labels)})
    if invalid_map_indices:
        preview = invalid_map_indices[:10]
        raise ValueError(
            f"Found out-of-range WordNet indices in map file. "
            f"Valid range: [0, {len(wn_labels) - 1}], examples: {preview}"
        )

    # Map COCO name -> WordNet index (0-9417)
    coco_name_to_idx = dict(zip(coco_names, coco_idxs))

    # Load annotations
    with open(path_in, 'r') as f:
        data = json.load(f)

    # 1. Filter images
    min_images = [
        {
            "id": img["id"],
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"]
        } for img in data.get('images', [])
    ]

    # Map COCO category_id -> Name (from COCO original dataset, e.g. 18 -> "dog")
    id_to_name = {c['id']: c['name'] for c in data.get('categories', [])}

    # 2. Filter annotations and Remap category_id
    min_annotations = []
    for ann in data.get('annotations', []):
        category_id = ann.get('category_id')
        if category_id is None:
            continue

        coco_name = id_to_name.get(category_id)
        if coco_name in coco_name_to_idx:
            wn_idx = coco_name_to_idx[coco_name]
            min_annotations.append({
                "image_id": ann["image_id"],
                "category_id": wn_idx,
                "bbox": ann["bbox"],
                "wordnet_id": wn_labels[wn_idx]
            })

    # 3. Create WordNet categories
    new_categories = [{"id": i, "name": label} for i, label in enumerate(wn_labels)]

    # Create new dictionary
    new_data = {
        "images": min_images,
        "annotations": min_annotations,
        "categories": new_categories
    }

    with open(path_out, 'w') as f:
        json.dump(new_data, f)
        
    print(f"Processed {len(min_annotations)} annotations out of {len(data.get('annotations', []))} original.")
    print(f"Saved to {path_out}")

def parse_args():
    parser = argparse.ArgumentParser(description='Minify COCO annotations and remap to WordNet.')
    parser.add_argument('-i', '--input-file', required=True, help='Path to the input COCO JSON.')
    parser.add_argument('-o', '--output-file', required=True, help='Path to output JSON.')
    parser.add_argument('--labels-file', default='9k.labels', help='Path to 9k.labels')
    parser.add_argument('--map-file', default='coco9k.map', help='Path to coco9k.map (with indices)')
    parser.add_argument('--names-file', default='coco.names', help='Path to coco.names (for class matching)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_dataset(args.input_file, args.output_file, args.labels_file, args.map_file, args.names_file)