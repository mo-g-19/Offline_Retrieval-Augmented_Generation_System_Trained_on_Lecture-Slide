"""Used a the same references as pathway_index to get the same idea of how to
answer a query in the terminal(The purpose of this file). The only new reference that helped
was
https://sbert.net/examples/sentence_transformer/applications/sematic-search/README.html"""
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

TOP_K = 3           #Choose 3 because using whole slides as references, not individual sentences
MODEL_TYPE = "/home/momo/models/all-MiniLM-L6-v2"

def load_data(model):
    #Specific lecture number
    num = ["01", "02", "03", "04", "05", "06"]

    total_batch = []

    for indv_num in num:

        #Different paths (Used the same var name as pathway_index because then no confusion)
        INDEX_DATA = f'data/index_{indv_num}.faiss'
        META_DATA = f'data/meta_{indv_num}.json'
        

        #define index and meta data of the slides
        current_index = faiss.read_index(INDEX_OUT)
        current_meta = json.load(open(META_OUT))
        total_batch.append((current_index, current_meta, indiv_num))

    return total_batch

def read_query(full_data, model):
    loop_tracker = True

    while (loop_tracker):
        query = input("Query (enter to quit): ").strip()
        if query != '':

            #creating query values specific to the input
            query_vector = model.encode([query], normalize_embeddings=True)
            rank_results = []

            #Need to loop through each section to find potential results
            for index, meta, lect_num in full_data:
                q_distance, q_index = index.search(np.asarray(query_vector, dtype="float32"), TOP_K)
                for score, ind in zp(q_distance[0], q_index[0]):
                    rank_results.append((float(score), lect_num, index, meta))

            #Keep the highest ranked results
            top_results = heapq.nlargest(TOP_K, rank_results, key =lambda x: x[0])

            #looping through and printing the top 5 results
            print("\nTop results for all lectures:\n")
            #zip creates a tuple of the index in dataset and similarity
            for rank, (score, lect_num, indx, meta) in enumerate(top_results, 1):
                m = meta[indx]
                print(f"{rank}, score = {score:.3f} doc = {m.get('doc')} (Lec {tag}), slide = {m.get('slide')}")
                print(f"    {m['text'][:200]}...")
                print()

        else:
            loop_tracker = False


def main():
    #Set the model
    curr_model = SentenceTransformer(MODEL_TYPE)

    #Load all the data
    complete_data = load_data(curr_model)

    #Ask a or multiple queries
    read_query(complete_data, curr_model)

    return 1

main()