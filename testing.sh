#!/bin/bash

echo "whoami: $(whoami)"
echo "Start time: $(date)"

#CHANGE THIS TO YOUR PATH
MODEL_PATH="/home/momo/models/all-MiniLM-L6-v2"
DATA_DIR="data"

# 6-shard run
echo "=== 6 Shards ==="
HF_HUB_OFFLINE=1 python quick_search.py \
    --model-path "$MODEL_PATH" \
    --data-dir "$DATA_DIR" \
    --lectures 01 02 03 04 05 06 \
    --evaluate eval.csv \
    --top-k 5 \
    #> eval_6shards.txt

# 3-shard group A
echo " "
echo "=== 3 Shards (Group A) ==="
HF_HUB_OFFLINE=1 python quick_search.py \
    --model-path "$MODEL_PATH" \
    --data-dir "$DATA_DIR" \
    --lectures 01 02 03 \
    --evaluate eval.csv \
    --top-k 5 \
    #> eval_3shards_A.txt

# 3-shard group B
echo " "
echo "=== 3 Shards (Group B) ==="
HF_HUB_OFFLINE=1 python quick_search.py \
    --model-path "$MODEL_PATH" \
    --data-dir "$DATA_DIR" \
    --lectures 04 05 06 \
    --evaluate eval.csv \
    --top-k 5 \
    #> eval_3shards_B.txt

echo "Finished at: $(date)"