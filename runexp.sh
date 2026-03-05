set -e
# =================================================================================
# DATASET creation
# =================================================================================

basedir=iNaturalist2021_val
datadir=$basedir/val
dataname=$basedir/"val.tar.gz"
metadata_name=$basedir/"val.json.tar.gz"

if [ -f "$dataname" -a -f "$metadata_name" ] ; then
    echo "dataset '$dataname' and metadata '$metadata_name' already present, skipping download"
else 
    # download the dataset and metadata
    echo "downloading dataset to '$datadir' folder"
    python scripts/dataset_download.py
fi
if [ -d "$datadir" ] ; then
    echo " '$datadir' folder already present, skipping unpacking"
else 
    # unpack the dataset and metadata
    echo "Unpacking '$dataname' to '$datadir' folder"
    tar -xvzf $dataname -C $basedir/
    tar -xvzf $metadata_name -C $basedir/
fi

# Create dataset
    dataset=dataset/
    rm -rf $dataset
    mkdir -p $dataset

    echo "Creating dataset and hierarchy from '$datadir' folder and '$metadata_name' metadata"

    python scripts/build_hierarchy.py \
    --val-json $basedir/val.json \
    --images-dir $basedir \
    --dataset-dir $dataset \
    --max-children 3 \
    --num-examples 5 \
    --output taxonomy_hierarchy \
    --kingdom Animalia \
    --inputs-dir inputs

    echo "Pre-computing DINOv2 embeddings for '$dataset'..."
    python scripts/embed_dataset.py $dataset

    echo "Dataset created in '$dataset' folder"

# =================================================================================
# EXPERIMENTS
# =================================================================================

PYTHONPATH=. python  scripts/fs2desc.py dataset descriptor.json
nexp=$(ls -1 inputs | wc -l) 
echo "loading $nexp experiments" 
counter=1 
mkdir -p results
 for i in inputs/*.json ; do 
     o=results/$(basename $i).npy.lz4
     if [ -f "$o" ] ; then
        echo [${counter}/${nexp}]: ${i} already done, skipping 
    else 
        echo -n [${counter}/${nexp}]": "
        PYTHONPATH=. python scripts/json_train.py  --results ${o} ${i} 
    fi 
    : $((counter++)) 
 done 
 echo creating figures... 

# =================================================================================
# FIGURES
# =================================================================================

tmpf=$(mktemp -d)
echo -n [1/3]": "
PYTHONPATH=. python  scripts/plot_hierarchy.py results/o_b_1.json.npy.lz4 results/a{95,90,80}_b_1.json.npy.lz4 --labels "full sup,a=0.95,a=0.90,a=0.80" --o ${tmpf}/semi >/dev/null
echo "done!"
echo -n [2/3]": "
PYTHONPATH=. python  scripts/plot_hierarchy.py  results/dummy.json.npy.lz4 results/o_b_1.json.npy.lz4 --labels "encounter, predict endounter" -o ${tmpf}/full >/dev/null
echo "done!"
echo -n [3/3]": "
PYTHONPATH=. python scripts/plot_hierarchy.py results/{a95_d,a90_b}_1.json.npy.lz4 --labels "devel,random"  -o ${tmpf}/setting >/dev/null
echo "done!"

echo "moving figures to outputs folder"
mkdir -p outputs
mv  ${tmpf}/{fullcost,semihf,semisup,settinghf,settingsup}.png outputs/
rm -r ${tmpf}