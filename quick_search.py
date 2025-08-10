"""Used a the same references as pathway_index to get the same idea of how to
answer a query in the terminal(The purpose of this file). The only new reference that helped
was
https://sbert.net/examples/sentence_transformer/applications/sematic-search/README.html"""
import os
import json
import heapq
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

TOP_K = 3           #Choose 3 because using whole slides as references, not individual sentences
PER_INDEX_K = 15
MODEL_TYPE = "/home/momo/models/all-MiniLM-L6-v2"
#Specific lecture number
LECTURES = ["01", "02", "03", "04", "05", "06"]

def read_index_pair(indv_slides):
    #Load data from Lectures
    data_files = [
        #Gives the flexibility if the slides are inside the file data instead of current file
        (f"data/index_{indv_slides}.faiss", f"data/meta_{indv_slides}.json"),
        (f"index_{indv_slides}.faiss", f"meta_{indv_slides}.json")
    ]

    for indx_path, meta_path in data_files:
        if os.path.exists(indx_path) and os.path.exists(meta_path):
            index = faiss.read_index(indx_path)
            with open(meta_path, "r") as file:
                meta = json.load(file)
            return index, meta


def load_data(model):
    total_batch = []

    for indv_num in LECTURES:


        #Making another function to open the specific data file and define index and meta data
        current_index, current_meta = read_index_pair(indv_num)
        
        if curr_index is not None and current_meta is not None:
            total_batch.append((current_index, current_meta, indv_num))

    return total_batch

def read_query(full_data, model):
    loop_tracker = True

    while (loop_tracker):
        query = input("Query (enter to quit): ").strip()
        if query != '':

            #creating query values specific to the input
            query_vector = model.encode([query], normalize_embeddings=True).astype("float32")
            rank_results = []

            #Need to loop through each section to find potential results
            for index, meta, lect_num in full_data:
                q_distance, q_index = index.search(query_vector, PER_INDEX_K)
                for score, idx in zip(q_distance[0], q_index[0]):
                    rank_results.append((float(score), lect_num, indx, meta))

            #Ensure unique slides with a dictionary
            for score, lect_num, idx, meta in rank_results:
                m = meta[idx]
                key = (m.get("doc"), m.get("slide"))
                if key not in best or score > best[key[0]]:
                    best[key] = (score, lect_num, m)

            #Keep the highest ranked results
            top_results = sorted(best.values(), key =lambda x: x[0], reverse=True)[:TOP_K]

            #looping through and printing the top 5 results
            print("\nTop results for all lectures:\n")
            #zip creates a tuple of the index in dataset and similarity
            for rank, (score, lect_num, m) in enumerate(top_results, 1):
                print(f"{rank}, score = {score:.3f} doc = {m.get('doc')} (Lec {lect_num}), slide = {m.get('slide')}")
                print(f"    {m.get('text', '')}...")
                print()

        else:
            loop_tracker = False


def main():
    #Set the model
    curr_model = SentenceTransformer(MODEL_TYPE)

    #Load all the data
    complete_data = load_data(curr_model)
    if not complete_data:
        print("No indexes loaded")
        return 1

    #Ask a or multiple queries
    read_query(complete_data, curr_model)
    return 0

main()