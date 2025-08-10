import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

#Specific lecture number
num = "01"

#Different paths (Used the same var name as pathway_index because then no confusion)
INDEX_OUT = "data/index_{num}.faiss"
META_OUT = "data/meta_{lnum}.json"
MODEL_TYPE = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3           #Choose 3 because using whole slides as references, not individual sentences

def read_query(index, meta, model):
    loop_tracker = True

    while (loop_tracker):
        query = input("Query (enter to quit): ").strip()
        if query != '':

            #creating query values specific to the input
            query_vector = model.encode([query], normalize_embeddings=True)
            q_distance, q_index = index.search(np.asarray(query_vector, dtype="float32"), TOP_K)

            #looping through and printing the top 5 results
            print("\Top results:\n")
            #zip creates a tuple of the index in dataset and similarity
            for rank, (ind, score) in enumerate(zip(q_index[0], q_distance[0]), 1):
                m = meta[ind]
                print(f"{rank}, score = {score:.3f} doc = {m.get('doc')}, page = {m.get('page')}")
                print(f"    {m['text'][:200]}...")
                print()

        else:
            loop_tracker = False


def main():
    #define index, meta, and model
    current_index = faiss.read_index(INDEX_OUT)
    curr_meta = json.load(open(META_OUT))
    curr_model = SentenceTransformer(MODEL_TYPE)

    read_query(current_index, curr_meta, curr_model)

    return 1

main()