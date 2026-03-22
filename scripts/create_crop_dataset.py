import os
import json
import uuid
import argparse
from tqdm import tqdm

def calculate_depth(obj):
    """
    Calcola la profondità di una gerarchia di liste annidate.
    Le stringhe (foglie) sono considerate a profondità 1.
    """
    if isinstance(obj, list):
        if not obj:
            return 1
        return 1 + max(calculate_depth(item) for item in obj)
    return 1

def main():
    parser = argparse.ArgumentParser(description="Estrae crop dalle annotazioni COCO e genera un dataset gerarchico.")
    parser.add_argument('-a', '--annotations', nargs='+', required=True, help='Percorsi ai file JSON delle annotazioni (es. instances_val2017_w.json)')
    parser.add_argument('-d', '--images-dirs', nargs='+', required=True, help='Percorsi alle cartelle delle immagini corrispondenti (es. val2017)')
    parser.add_argument('-m', '--metadata', required=True, help='Percorso al file metadata.json')
    parser.add_argument('-o', '--out-dir', default='dataset', help='Cartella in cui salvare i crop')
    parser.add_argument('--desc', default='descriptor_out.json', help='Percorso del file descriptor in output')
    args = parser.parse_args()

    if len(args.annotations) != len(args.images_dirs):
        print("Errore: il numero di file di annotazione deve coincidere con il numero di cartelle immagini.")
        return

    # Import ritardato per non rallentare l'avvio della logica di base
    from PIL import Image
    from collections import defaultdict
    
    # Caricamento hierarchy
    with open(args.metadata, 'r') as f:
        metadata = json.load(f)
        
    hierarchy = metadata.get("hierarchy", [])
    depth = calculate_depth(hierarchy)
    
    os.makedirs(args.out_dir, exist_ok=True)
    classes_paths = defaultdict(list)
    crop_errors = 0
    max_error_logs = 20
    
    # Elaborazione
    for ann_file, img_dir in zip(args.annotations, args.images_dirs):
        with open(ann_file, 'r') as f:
            data = json.load(f)
            
        img_dict = {img['id']: img['file_name'] for img in data.get('images', [])}
        annotations = data.get('annotations', [])
        
        for ann in tqdm(annotations, desc=f"Elaborazione {os.path.basename(ann_file)}"):
            img_id = ann.get('image_id')
            if img_id not in img_dict:
                continue
                
            img_path = os.path.join(img_dir, img_dict[img_id])
            if not os.path.exists(img_path):
                continue
                
            # Recuperiamo la classe/nome per la cartella
            class_name = ann.get('wordnet_id')
            if not class_name:
                continue
                
            bbox = ann.get('bbox')
            if not bbox or len(bbox) != 4:
                continue
                
            x, y, w, h = [int(v) for v in bbox]
            if w <= 0 or h <= 0:
                continue
                
            class_dir = os.path.join(args.out_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Nuovo filename con UUID (formato jpg)
            new_filename = f"{uuid.uuid4()}.jpg"
            out_img_path = os.path.join(class_dir, new_filename)
            
            try:
                # Estrazione e crop
                with Image.open(img_path) as im:
                    crop = im.crop((x, y, x + w, y + h))
                    if crop.mode in ("RGBA", "P"):
                        crop = crop.convert("RGB")
                    crop.save(out_img_path, format="JPEG")
                    
                # Path relativo POSIX
                rel_path = os.path.join(args.out_dir, class_name, new_filename).replace('\\', '/')
                classes_paths[class_name].append(rel_path)
            except Exception as e:
                crop_errors += 1
                if crop_errors <= max_error_logs:
                    print(f"[WARN] Crop failed for {img_path} (bbox={bbox}): {e}")

    if crop_errors > max_error_logs:
        print(f"[WARN] Additional crop errors suppressed: {crop_errors - max_error_logs}")

    # Salvataggio descrittore finale
    descriptor_classes = []
    for idx, class_name in enumerate(sorted(classes_paths.keys())):
        descriptor_classes.append({
            "id": idx,
            "name": class_name,
            "paths": sorted(classes_paths[class_name])
        })
        
    descriptor = [
        {
            "hierarchy": hierarchy,
            "depth": depth
        },
        descriptor_classes
    ]
    
    with open(args.desc, 'w') as f:
        json.dump(descriptor, f, indent=4)
        
    tot_crops = sum(len(p) for p in classes_paths.values())
    print(f"\nOperazione completata!")
    print(f"I crop ({tot_crops}) sono stati salvati in: {args.out_dir}")
    if crop_errors:
        print(f"Crop skipped due to errors: {crop_errors}")
    print(f"File descriptor salvato come: {args.desc}")

if __name__ == "__main__":
    main()