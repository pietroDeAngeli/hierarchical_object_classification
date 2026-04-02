set -e
TEST_SIZE=140
TEST_SEED=0
TEST_SPLIT_DIR=${TEST_SPLIT_DIR:-results/test_splits}

# Usage:
#   ./runexp.sh                         # default: all inputs/*.json
#   ./runexp.sh inputs/a80_b_1.json     # single input file
#   ./runexp.sh inputs/a80_b_1.json inputs/o_b_1.json
if [ "$#" -gt 0 ] ; then
    INPUT_FILES=("$@")
else
    shopt -s nullglob
    INPUT_FILES=(inputs/*.json)
    shopt -u nullglob
fi

if [ "${#INPUT_FILES[@]}" -eq 0 ] ; then
    echo "No input JSON files found."
fi


# =================================================================================
# DATASET creation
# =================================================================================


# =================================================================================
# EXPERIMENTS
# =================================================================================

PYTHONPATH=. python  scripts/fs2desc.py dataset descriptor.json
nexp=${#INPUT_FILES[@]}
echo "loading $nexp experiments" 
counter=1 
mkdir -p results
if [ "$TEST_SIZE" -gt 0 ] ; then
    mkdir -p "$TEST_SPLIT_DIR"
    echo "fixed test split enabled: TEST_SIZE=$TEST_SIZE TEST_SEED=$TEST_SEED"
fi
for i in "${INPUT_FILES[@]}" ; do 
      if [ ! -f "$i" ] ; then
          echo [${counter}/${nexp}]: ${i} not found, skipping
          : $((counter++))
          continue
      fi
     o=results/$(basename $i).npy.lz4
     if [ -f "$o" ] ; then
        echo [${counter}/${nexp}]: ${i} already done, skipping 
    else 
        echo -n [${counter}/${nexp}]": "
        if [ "$TEST_SIZE" -gt 0 ] ; then
            test_json=${TEST_SPLIT_DIR}/$(basename "$i" .json)_test.json
            PYTHONPATH=. python scripts/json_train.py --results ${o} ${i} \
                --test-size "$TEST_SIZE" --test-seed "$TEST_SEED" --test-output "$test_json" --eval-test
        else
            PYTHONPATH=. python scripts/json_train.py --results ${o} ${i}
        fi
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