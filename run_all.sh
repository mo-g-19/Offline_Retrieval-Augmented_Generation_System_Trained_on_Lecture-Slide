#!/bin/bash
set -e

MODEL_PATH="/home/momo/models/all-MiniLM-L6-v2"
DATA_DIR="data"

echo "=== Step 1: Process raw slides into processed.json ==="
HF_HUB_OFFLINE=1 python readSlides.py --data-dir "$DATA_DIR"

echo "=== Step 2: Build FAISS indexes for all 6 lectures ==="
for i in 01 02 03 04 05 06; do
    HF_HUB_OFFLINE=1 python pathway_index.py \
        --model-path "$MODEL_PATH" \
        --input-json "$DATA_DIR/lecture${i}_processed.json" \
        --out-dir "$DATA_DIR"
done

echo "=== Step 3: Test quick_search.py ==="
HF_HUB_OFFLINE=1 python quick_search.py \
    --data-dir "$DATA_DIR" \
    --lectures 01 02 03 04 05 06 \
    --evaluate eval.csv \
    --top-k 5