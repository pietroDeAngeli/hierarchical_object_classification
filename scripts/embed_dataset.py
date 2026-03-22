"""
Converts raw images in a dataset folder to pre-computed DINOv2 CLS embeddings.
Expects the folder structure:  dataset/<class_name>/<image_file>
Replaces each .jpg/.jpeg/.png with a .npy embedding file of the same base name.
The resulting folder can be used with "pre_embedded": true in the experiment configs.
"""
import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_MODEL = "facebook/dinov2-small"
DEFAULT_BATCH_SIZE = 32


def get_all_images(dataset_dir, descriptor_path=None):
    """Returns list of image paths for all raw images in the dataset.
    If descriptor_path is provided, only processes images listed in the descriptor."""
    items = []
    
    if descriptor_path and os.path.exists(descriptor_path):
        with open(descriptor_path, 'r') as f:
            descriptor = json.load(f)
        
        # Structure is [meta, [classes]]
        if len(descriptor) >= 2 and isinstance(descriptor[1], list):
            for cls_data in descriptor[1]:
                for path_str in cls_data.get('paths', []):
                    path = Path(path_str)
                    if path.suffix.lower() in IMAGE_EXTENSIONS:
                        items.append(path)
        return items

    dataset_dir = Path(dataset_dir)
    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append(img_path)
    return items


def get_pending_images(dataset_dir, keep_images, descriptor_path=None):
    """Returns only images that still need to be embedded (no .npy counterpart yet).
    Also cleans up orphan images (npy exists but jpg remains) when keep_images=False."""
    pending = []
    for img_path in get_all_images(dataset_dir, descriptor_path):
        npy_path = img_path.with_suffix(".npy")
        if npy_path.exists():
            if not keep_images and img_path.exists():
                img_path.unlink(missing_ok=True)  # clean up leftover image
        else:
            pending.append(img_path)
    return pending


def embed_dataset(dataset_dir, model_name, batch_size, keep_images, descriptor_path=None):
    image_paths = get_pending_images(dataset_dir, keep_images, descriptor_path)
    if not image_paths:
        print("Nothing to embed — all images already have .npy counterparts.")
        if descriptor_path:
            create_embedded_descriptor(descriptor_path)
        return

    print(f"Found {len(image_paths)} images to embed.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model {model_name}...")

    processor = AutoImageProcessor.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModel.from_pretrained(model_name, token=HF_TOKEN).to(device)
    model.eval()

    batch_paths = []
    batch_images = []

    def flush_batch():
        if not batch_images:
            return
        inputs = processor(images=batch_images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        for path, emb in zip(batch_paths, cls_embeddings):
            npy_path = path.with_suffix(".npy")
            np.save(str(npy_path), emb)
            if not keep_images:
                path.unlink()
        batch_paths.clear()
        batch_images.clear()

    for img_path in tqdm(image_paths, desc="Embedding images"):
        try:
            pil_image = Image.open(img_path).convert("RGB")
            img_array = np.array(pil_image)
            
            # Filtro per immagini anomale/corrotte (troppo piccole o con canali errati)
            if len(img_array.shape) != 3 or img_array.shape[-1] != 3 or img_array.shape[0] < 5 or img_array.shape[1] < 5:
                print(f"  [WARN] Skipping corrupted/tiny image: {img_path} with shape {img_array.shape}")
                continue
                
        except Exception as e:
            print(f"  [WARN] Cannot open {img_path}: {e}")
            continue

        batch_paths.append(img_path)
        batch_images.append(pil_image)

        if len(batch_images) >= batch_size:
            flush_batch()

    flush_batch()

    print(f"\nDone. Embeddings saved.")

    # Convert the dataset descriptor if requested
    if descriptor_path:
        create_embedded_descriptor(descriptor_path)

def create_embedded_descriptor(descriptor_path):
    print(f"Creating embedded descriptor from {descriptor_path}...")
    with open(descriptor_path, 'r') as f:
        descriptor = json.load(f)
        
    if len(descriptor) < 2 or not isinstance(descriptor[1], list):
        print("Descriptor format not recognized. Skipping creation of embedded descriptor.")
        return

    # Keep only entries that have a computed embedding.
    removed_paths = 0
    for cls_data in descriptor[1]:
        if 'paths' in cls_data:
            updated_paths = []
            for p in cls_data['paths']:
                path = Path(p)
                npy_path = path.with_suffix(".npy")
                if path.suffix.lower() in IMAGE_EXTENSIONS and npy_path.exists():
                    updated_paths.append(str(npy_path))
                elif path.suffix.lower() == ".npy" and path.exists():
                    updated_paths.append(str(path))
                else:
                    removed_paths += 1
                    print(f"  [WARN] Embedding not found, removing from descriptor: {p}")
            
            cls_data['paths'] = updated_paths

    out_path = Path(descriptor_path)
    out_path = out_path.with_name(f"{out_path.stem}_embedded{out_path.suffix}")
    
    with open(out_path, 'w') as f:
        json.dump(descriptor, f, indent=2)
        
    print(f"Embedded descriptor saved to {out_path}")
    if removed_paths:
        print(f"Removed {removed_paths} paths without computed embeddings.")

def main():
    parser = argparse.ArgumentParser(description="Pre-compute DINOv2 embeddings for a dataset folder.")
    parser.add_argument("dataset_dir", type=str, nargs='?', default='dataset',
                        help="Root dataset directory (structure: dataset/<class>/<images>)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"HuggingFace model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size for inference (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--keep-images", action="store_true",
                        help="Keep original image files after embedding (default: delete them)")
    parser.add_argument("--descriptor", type=str,
                        help="Path to the JSON descriptor generated by create_crop_dataset.py")
    args = parser.parse_args()

    embed_dataset(args.dataset_dir, args.model, args.batch_size, args.keep_images, args.descriptor)

if __name__ == "__main__":
    main()