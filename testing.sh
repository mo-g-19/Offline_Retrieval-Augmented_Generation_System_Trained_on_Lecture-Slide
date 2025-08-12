#!/bin/bash

echo "whoami: $(whoami)"
echo "Start time: $(date)"

# 6-shard run
echo "--- 6 Shards ---"
time HF_HUB_OFFLINE=1 python quick_search.py \
    --data-dir data \
    --lectures 01 02 03 04 05 06 \
    --evaluate eval.csv \
    --top-k 5 > \
    > eval_6shards.txt

#3 Shards - first 3 slides
echo "--- 3 Shards (Group A) ---"
time HF_HUB_OFFLINE=1 python quick_search.py \
    --data-dir data \
    --lectures 01 02 03 \
    --evaluate eval.csv \
    --top-k 5 \
    > eval_3shards_A.txt

#Second 3 shards (Group B)
echo "--- 3 Shards (Group B) ---"
time HF_HUB_OFFLINE=1 python quick_search.py \
    --data-dir data \
    --lectures 04 05 06 \
    --evaluate eval.csv \
    --top-k 5 \
    > eval_3shards_B.txt

echo "Finished at: $(date)"