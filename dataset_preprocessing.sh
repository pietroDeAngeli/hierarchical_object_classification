#!/usr/bin/env bash
set -euo pipefail

selected_partition="val"
# selected_partition="train"


wordnet_hierarchy="9k.tree"
wordnet_labels="9k.labels"
wordnet_map="coco9k.map"
coco_names="coco.names"
annotations_dir="annotations"


if [ -d "dataset" ]; then
    echo "Dataset already present, skipping creation"
else
    # Default annotation file
    if [ "$selected_partition" = "val" ]; then
        input_file="$annotations_dir/instances_val2017.json"
        out="$annotations_dir/instances_val2017_w.json"
    else
        input_file="$annotations_dir/instances_train2017.json"
        out="$annotations_dir/instances_train2017_w.json"
    fi

    # Annotation download and unpacking
    if [ -d "$annotations_dir" ]; then
        echo "TrainVal Annotations already present, skipping download and unpacking"
    else
        if [ -f annotations_trainval2017.zip ]; then
            echo "TrainVal Annotations zip already present, skipping download"
        else
            echo "Downloading TrainVal Annotations"
            wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
        fi
        unzip annotations_trainval2017.zip -d .
        rm -f "$annotations_dir"/captions_* "$annotations_dir"/person_keypoints_*
    fi

    # Download mapping coco -> wordnet
    if [ -f "$wordnet_map" ]; then
        echo "WordNet map file already present, skipping download"
    else
        wget https://raw.githubusercontent.com/pjreddie/darknet/1e729804f61c8627eb257fba8b83f74e04945db7/data/coco9k.map
        echo "WordNet map downloaded"
    fi

    # Download COCO names for mapping
    if [ -f "$coco_names" ]; then
        echo "COCO names file already present, skipping download"
    else
        wget https://raw.githubusercontent.com/pjreddie/darknet/1e729804f61c8627eb257fba8b83f74e04945db7/data/coco.names
        echo "COCO names downloaded"
    fi

    # Download WordNet labels
    if [ -f "$wordnet_labels" ]; then
        echo "WordNet labels file already present, skipping download"
    else
        wget https://raw.githubusercontent.com/pjreddie/darknet/1e729804f61c8627eb257fba8b83f74e04945db7/data/9k.labels
        echo "WordNet labels downloaded"
    fi

    # Hierarchy creation
    if [ -f "metadata.json" ]; then
        echo "Metadata already present, skipping hierarchy creation"
    else
        if [ -f "$wordnet_hierarchy" ]; then
            echo "WordNet hierarchy file already present, skipping download"
        else
            wget https://raw.githubusercontent.com/pjreddie/darknet/1e729804f61c8627eb257fba8b83f74e04945db7/data/9k.tree
            echo "Wordnet hierarchy downloaded"
        fi

        python scripts/generate_metadata.py
    fi
    
    # Process annotations
    if [ -f "$out" ]; then
        echo "Processed annotations already present, skipping processing"
    else
        
        python scripts/process_annotations.py \
            -i "$input_file" \
            -o "$out" \
            --labels-file "$wordnet_labels" \
            --map-file "$wordnet_map" \
            --names-file "$coco_names"
    fi

    if [ "$selected_partition" = "val" ]; then
        if [ -d "val2017" ]; then
            echo "Validation images already present, skipping download"
        else
            if [ -f val2017.zip ]; then
                echo "Validation images zip already present, skipping download"
            else
                echo "Downloading COCO Validation set"
                wget http://images.cocodataset.org/zips/val2017.zip
            fi
            unzip val2017.zip -d .
        fi
    else
        if [ -d "train2017" ]; then
            echo "Training images already present, skipping download"
        else
            if [ -f train2017.zip ]; then
                echo "Training images zip already present, skipping download"
            else
                echo "Downloading COCO train set"
                wget http://images.cocodataset.org/zips/train2017.zip
            fi
            unzip train2017.zip -d .
        fi
    fi

    input_file=$out

    python scripts/create_crop_dataset.py \
        -a "$input_file" \
        -d "${selected_partition}2017" \
        -m metadata.json \
        -o dataset/ \
        --desc descriptor.json

    echo "Dataset creation completed."
fi

# Check if images are not embeeded, if they are not calculate the embeddings
if [ -n "$(find dataset/ -type f \( -name '*.jpg' -o -name '*.png' \) -print -quit 2>/dev/null)" ]; then
    echo "Found unprocessed images. Calculating embeddings..."
    # To keep original images, add the flag --keep-images below
    python scripts/embed_dataset.py \
        dataset/ \
        --model "facebook/dinov2-base" \
        --batch-size 32 \
        --descriptor descriptor.json
    echo "Embeddings calculation completed."
else
    echo "Embeddings already calculated (no images found)."
fi

mv metadata.json dataset/
find dataset -name "*.jpg" -type f -delete
