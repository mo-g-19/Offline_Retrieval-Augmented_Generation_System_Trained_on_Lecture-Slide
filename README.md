# Offline Retrieval-Augmented Generation System Trained on Lecture Slide
This project implements a two-phase Retrieval-Augmented Generation (RAG) pipeline using FAISS for vector search and a Sentence Transformer model (from HuggingFace) for embeddings. It supports both single-node and multi-shard execution for performance analysis.

## Files
    home
        models/ # Holds the objects of Sentence Transformer
            all-MiniLM-L6-v2/ # The specific folder that holds the Sentence Transformer
        data/ #Directory for raw slides, processed JSON, FAISS indexes, metadata
            raw_slides/ # Original lecture PDF slides
            lecture##_processed.json
            index_##.faiss
            meta_##.json
        readSlides.py # Extracts text from PDF slides into JSON
        pathway_index.py # Builds FAISS indexes from processed JSON
        quick_search.py # Runs queries against one or more FAISS indexes
        eval.csv #Evaluation queries and expected answers
        eval_#shards #My results from running eval.csv
        testing_all_python_files.sh # Turns the pdf slides into metadata and 384 dimensional FAISS vectors
        testing.sh # Evaluates and prints out the latency and accuracy depending on the shard number
        README.md # This file

## Needed Libraries and Model
- Python 3.10+
- [PyMuPDF] (https://pymupdf.readthedocs.io/)
- [PyInstaller] (https://pyinstaller.org/en/stable/)
- [FAISS] (https://githumb.com/facebookreasearch/faiss)
- [SentenceTransformers] (https://www.huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Torch] (https://pytorch.org/)


## Requirements to Set Up The System
# Create and activate a virtual enviroment
'''python3 -m venv venv
source venv/bin/activate'''

# Install libraries and snapshot
'''pip install python3 pyinstaller
pip install python3 toch sentence-transformers faiss-cpu pymupdf'''

# Get the Text from PDF of Slides
You can change the testing_all_python_files.sh to your files, but here are the commands if you want to do it yourself

Repeat the following steps with 01 for 02 03 04 05 06
'''HF_HUB_OFFLINE=1 python readSlides.py \
    --input-pdf ./data/raw_slides/lecture01.pdf \
    --output-json ./data/lecture01_processed.json'''

'''HF_HUB_OFFLINE=1 python pathway_index.py \
        --model-path ./data \
        --input-json ./data/lecture$01_processed.json \
        --output-index ./data/index_01.faiss \
        --output-meta ./data/meta_01.json \
        --out-dir ./data'''

To make your own query
'''HF_HUB_OFFLINE=1 python quick_search.py \
    --data-dir ./data \
    --lectures 01 02 03 04 05 06 \
    --model-path /home/models/all-MiniLM-L6-v2 \
    --query "What is the OSI model?"'''

To run the evaluation I did either:
'''HF_HUB_OFFLINE=1 python quick_search.py \
    --data-dir ./data \
    --lectures 01 02 03 04 05 06 \
    --evaluate eval.csv
    --top-k 5'''

or
'''bash run_all.sh'''

# Output Files
Check the eval_#shards*.txt to see the results I saved

# References
Loading sections and embedding
1) Sphinx: https://sbert.net/examples/applications/computing-embeddings/docs/docs/docs/sentence_transformer/usage/examples/sentence_transormer/applications/computing-embeddings/README.html
2) Sphinx: https://sbert.net/docs/package_reference/sentence_transformer/SentanceTransformer.html#sentence_transformers.SentenceTransformer.encode

FAISS indexing and why chose cosine 
3) G. Wang: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
4) Matthjis Douze: https://github.com/facebookresearch/faiss/wiki/Getting-started

Saving the indexing
5) Mathjis Douze : https://github.com/facebookresearch/faiss/wiki/IO,-index-factory,-and-metadata#io