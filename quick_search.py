"""Used a the same references as pathway_index to get the same idea of how to
answer a query in the terminal(The purpose of this file). The only new reference that helped
was
https://sbert.net/examples/sentence_transformer/applications/sematic-search/README.html"""
import os
import json
import argparse     #Used to add a flag to find the index and meta files of the lectures incase it is no longer in ./data
#import heapq       #realize I don't need it anymore
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

#Forcing the enviroment to be offline (this VC isn't connected to the internet)
os.environ["HF_HUB_OFFLINE"] = 1

TOP_K = 5           #Changed to 5 because too few results
PER_INDEX_K = 15
MODEL_TYPE = "/home/momo/models/all-MiniLM-L6-v2"
#Specific lecture number
LECTURES = ["01", "02", "03", "04", "05", "06"]

def read_index_pair(lecture_num, data_dir):
    """Loading the slides data from index and meta.
    Ensures there is a pathway to access said data"""

    #Load data from Lectures, first checks the data directory
    index_path = os.path.join(data_dir, f"index_{lecture_num}.faiss")
    meta_path = os.path.join(data_dir, f"meta_{lecture_num}.json")
    if os.path.exists(index_path) and os.path.exists(meta_path):
        return faiss.read_index(index_path), json.load(open(meta_path, "r"),), ("data", index_path, meta_path)

    #Then check the current directory
    index_path = f"index_{lecture_num}.faiss"
    meta_path = f"meta_{lecture_num}.json"
    if os.path.exists(index_path) and os.path.exists(meta_path):
        return faiss.read_index(index_path), json.load(open(meta_path, "r")), (".", index_path, meta_path)

    #Else, return nothing
    return None, None, None

    """data_files = [
        #Gives the flexibility if the slides are inside the file data instead of current file
        (f"data/index_{indv_slides}.faiss", f"data/meta_{indv_slides}.json"),
        (f"index_{indv_slides}.faiss", f"meta_{indv_slides}.json")
    ]

    for indx_path, meta_path in data_files:
        if os.path.exists(indx_path) and os.path.exists(meta_path):
            index = faiss.read_index(indx_path)
            with open(meta_path, "r") as file:
                meta = json.load(file)
            return index, meta"""


def load_data(lectures, data_dir):
    """ Load all the index, meta, and lecture number data into triples, and print a debug statement of what happened"""

    total_batch = []

    for indv_num in lectures:
        #Making another function to open the specific data file and define index and meta data
        index, meta, tuple_path = read_index_pair(indv_num)
        #Only allows to save info if there is a vector and meta data attached to the slides
        if index is not None and meta is not None:
            total_batch.append((idx, meta, num, tuple_path))
    
    #Print statement if there are no indexes
    if not total_batch:
        print("No indexes loaded.")
        print(f"Looked for files like index_XX.faiss and meta_XX.json in '{data_dir}' and the current directory")
        return []
    
    print(f"Loaded {len(batch)} lecture inexes:")
    for index, meta, indv_num, tuple_path in total_batch:
        location, index_path, meta_path = tuple_path
        match_check = (idx.ntotal == len(meta))
        print(f"    lec {num}: vectors={index.ntotal}   meta={len(meta)}    match={match_check}     from={location}  ({index_path}, {meta_path})")
    return total_batch
        
    """    if index is not None and meta is not None:
            total_batch.append((index, meta, indv_num))

    return total_batch"""

def read_query(model, full_data_batch, query, per_index_k, top_k):
    """Read the query, find each index, and return with the top_k results"""
    #loop_tracker = True

    #while (loop_tracker):
    #query = input("Query (enter to quit): ").strip()
        #if query != '':

    #creating query values specific to the input
    query_vector = model.encode([query], normalize_embeddings=True).astype("float32")
    
    rank_results = []

    #Need to loop through each section to find potential results
    for index, meta, lect_num in full_data_batch:
        q_distance, q_index = index.search(query_vector, PER_INDEX_K)
        for score, idx in zip(q_distance[0], q_index[0]):
            rank_results.append((float(score), lect_num, meta[int(idx)]))

    #Enure unique slides with a dictionary while keeping the best score
    best = {}
    for score, lect_num, meta in rank_results:
        key = (meta.get("doc"), meta.get("slide"))
        if key not in best or score > best[key][0]:
            best[key] = (score, lect_num, meta)

    #Keep the highest ranked results
    top_results = sorted(best.values(), key =lambda x: x[0], reverse=True)[:TOP_K]
    
    #New tuple that returns the top_k results
    final_ranked = sorted(best.values(), key=lambda x: x[0], reverse=True)[:top_k]
    return final_ranked

def main():
    #Creating arguments for loading the data; used docs.python.org/3/library/argparse.html documentation as a guide and why I chose using arguments
        #Reason: I needed something to print out the global var and file paths, and this was easier than trying to use variable names and rewritting directory names
    ap = argparse.ArgumentParser(description = "Offline semantic search over lecture indexes (FAISS).")
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local SentenceTransformer model path")
    ap.add_argument("--lectures", nargs="+", default=DEFAULT_LECTURES, help="Lecture numbers to load, e.g. 01, 02, 03, 04...")
    ap.add_argument("--data-dir", default="data", help="Directory containing index_XX.faiss and meta_XX.json")
    ap.add_argument("--per-index-k", type=int, default=PER_INDEX_K_DEFAULT, help="Per-index candidates (before merge)")
    ap.add_argument("--top-k", type=int, default=TOP_K_DEFAULT, help="Final merged top-k results")
    ap.add_argument("--print-chars", type=int, default=500, help="Chars to print from each hit (<=0 prints full text)")
    args = ap.parse_args()

    #Set the model and print out that the model is loaded offline locally
    print(f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')} model={args.model_path}")
    curr_model = SentenceTransformer(MODEL_TYPE)

    #Load all the data (index/meta)
    current_data = load_data(args.lectures, args.data_dir)
    if not current_data:
        print("No indexes loaded")
        raise SystemExit(2)

    #The interactive loop that moved from read_query funct
    active_query = TRUE
    while active_query:
        #Using a try, if not error statement to ensure a safe recovery if end of file error or user presses enter
        try:
            current_querry = input("Query (enter to quit): ").strip()
        except EOFError:
            break
        if not current_querry:
            break
        
        #Running and evaluating the querry
        current_ranked = run_query(model, current_data, current_querry, args.per_index_k, args.top_k)

        print("\nTop results for all lectures:\n")
        if not ranked:
            print("No results found from this query")
        else:
            for ranked_results, (score, lect_num, m) in enumerate(ranked, 1):
                snippet = m.get("text", "")
                if args.print_chars > 0:
                    snippet = snippet[:args.print_chars]
                print(f"{ranked_results}. score={score:.3f}  doc={m.get('doc')   (Lec {lect_num}),  slide={m.get('slide')}}")
                print(f"    {snippet}\n")


"""
    #Ask a or multiple queries
    read_query(complete_data, curr_model)
    return 0"""

main()