#!/bin/bash

#Exit if any command fails
set -e 

#CHANGE THIS TO FIT YOUR PATH SET UP
MODEL_PATH="/home/momo/models/all-MiniLM-L6-v2"
#CHANGE IF KEEPING lecture##_processed, index_, or meta_ are in another directory
DATA_DIR="$(pwd)"

echo "--- Converting the pdf slides into metadata and FAISS vectors"
echo ""

#CHANGE 'input-pdf' if the PDF SLIDES ARE IN A DIFFERENT DIRECTORY THAN INSIDE 'data' in a directory called 'raw_slides'
echo "--- Step 1: Run readSlides.py ---"
for i in 01 02 03 04 05 06; do
    HF_HUB_OFFLINE=1 python readSlides.py \
        --input-pdf "$DATA_DIR/raw_slides/lecture${i}.pdf" \
        --output-json "$DATA_DIR/lecture${i}_processed.json"
done

echo ""

echo "--- Step 2: Run pathway_index.py ---"
for num in 01 02 03 04 05 06; do
    HF_HUB_OFFLINE=1 python pathway_index.py \
        --model-path "$MODEL_PATH" \
        --input-json "$DATA_DIR/lecture${num}_processed.json" \
        --output-index "$DATA_DIR/index_${i}.faiss" \
        --output-meta "$DATA_DIR/meta_${i}.json" \
        --out-dir "$DATA_DIR"

done

echo ""
echo "--- Done ---"
