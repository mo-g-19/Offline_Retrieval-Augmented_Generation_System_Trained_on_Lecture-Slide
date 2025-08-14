"""The purpose of this program is to create the embeddings from the files 
and create a FAISS index that will be saved in a seperate file. I will list the
the other websites I used as references and helped me build my work

Loading sections and embedding
1) Sphinx: https://sbert.net/examples/applications/computing-embeddings/docs/docs/docs/sentence_transformer/usage/examples/sentence_transormer/applications/computing-embeddings/README.html
2) Sphinx: https://sbert.net/docs/package_reference/sentence_transformer/SentanceTransformer.html#sentence_transformers.SentenceTransformer.encode

FAISS indexing and why chose cosine 
3) G. Wang: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
4) Matthjis Douze: https://github.com/facebookresearch/faiss/wiki/Getting-started

Saving the indexing
5) Mathjis Douze : https://github.com/facebookresearch/faiss/wiki/IO,-index-factory,-and-metadata#io
"""

import os
import json
import numpy as np
import faiss
#import glob
import argparse
import re
from sentence_transformers import SentenceTransformer

#sentence_transformer offline
os.environ["HF_HUB_OFFLINE"] = "1"


#Load Sections
def creating_sections(processed_path, lecture_num):
    """Purpose: To take the slides and make it easier to create a FAISS index vector and organize the metadata
    Input: The file path to the text file that is an array full of dictionaries of slides and the number of the presentation
    Output: An array of the full text and an array of dictionaries with the doc, slide, and text
    """
    current_sections = []
    full_text = []
    meta_data = []
    
    #open the file and for each line, copy onto "current_sections" so the running list is on file
    with open(processed_path, "r") as proccessed_file:
        current_sections = json.load(proccessed_file)
    
    for page_num, text_count in enumerate(current_sections):
        out_of_context_text = (text_count.get("text")).strip()
        
        #adding to the full_text
        full_text.append(out_of_context_text)

        #Making a 4-tuple that 
        meta_data.append({
            "doc" : lecture_num,
            "slide" : text_count.get("id"),
            "text" : out_of_context_text[:500]
        })

    #Check that the text is successfully loaded
    print(f"Loaded {len(full_text)} sections")

    return full_text, meta_data



#Embed - Start using references
def embed_text(full_text, current_model):
    """Purpose: Take the entire paragraph of text and turn it into a dense vector representation
    Input: The text of a slide and an object of the SentenceTransformer
    Output: A list of embeddings (An array that holds an array with 32 float values); 384 (see on huggingface for sentence-transformers/all-MiniLM-L6-v2) dimensional dense vector space
    """
    
    #Using 2 for lines 66-74
    #Using encode because want the most general method and will have a "model [that] was not trained with pedefined prompts and/or task types"
    
    #batch_size -> 64 (normally 32): because I have limited RAM and had to completely start over because I tried to clear space and did too much clearing
    #show_progress_bar -> True: Want to see it when encoding sentances (seemed like a cool feature)
    #Normalize_embeddings -> True; because want to keep on a 0-1 range (important for FAISS)
    embeddings = current_model.encode(full_text, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    #Making embeddings for sure an ndarray
    embeddings = np.asarray(embeddings, dtype="float32")

    #Confirm that successfully embed
    print("Successfully embedded the full text array")
    return embeddings
    

#Build FAISS index (Using IndexFlatIP because it is a small data set and no need for train/tune) - reference 3
def build_faiss_index(embeddings):
    """Purpose: Create an index vector (FAISS) (eventually used to find top-k similar vectors without scan entire dataset)
    Input: The dense vector representation (embed_text's return)
    Output: FAISS index opject in memory for immediate use (does write FAISS index file to disk)
    """
    #Find the number of columns needed for the indexing
    embeddings_dimension = embeddings.shape[1]

    #Reference 4 for lines 87-91
    #Using cosine similarity to see how "similiar/same direction"; unlike the example given in 4
    #index => "encapsulates the set of database vecotrs, and optionally preprocesses them to make searching efficient"
    index = faiss.IndexFlatIP(embeddings_dimension)
    #adding the ids to the indexed vectors
    index.add(embeddings)

    #Confirm that it did index
    print(f"Indexed {index.ntotal} vectors (dimension = {embeddings_dimension})")

    return index

#Main function and save file
def main():
    """Loades the slides from the lecture, embed them, build and save a FAISS index, and save the metadata
    """

    #Takes command prompts to store as paths
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="./models/all-MiniLM-L6-v2")
    ap.add_argument("--input-json", required=True, help="Path to lecture##_process.json")
    ap.add_argument("--output-index", default="./Desktop/RAG/data")
    ap.add_argument("--output-meta", default="./Desktop/RAG/data")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    #Saving the arguments as variables
    model_type = args.model_path
    processed_path = args.input_json

    #Finding the number based on the processed_path
    derive_num = re.search(r"lecture(\d+)_processed\.json$", os.path.basename(processed_path))
    num = derive_num.group(1) if derive_num else "00"
    os.makedirs(args.out_dir, exist_ok=True)
    index_out = os.path.join(args.out_dir, f"index_{num}.faiss")
    meta_out = os.path.join(args.out_dir, f"meta_{num}.json")
    
    #Loading the model and settings - specifically "used to map sentences/text to embeddings"
    current_model = SentenceTransformer(model_type)         #saved specifically locally

    #global var that will become var in main
    text = []  #full text that gets referenced by meta
    data = []  #data that will get loaded

    text, data = creating_sections(processed_path, num)
    curr_embed = embed_text(text, current_model)
    curr_index = build_faiss_index(curr_embed)

    #Save to file
    faiss.write_index(curr_index, index_out)
    with open(meta_out, "w") as file:
        json.dump(data, file, indent = 2)

    #Confirm the save
    print(f"Saved index -> {os.path.abspath(index_out)}")
    print(f"Saved meta  -> {os.path.abspath(meta_out)}")

    return

main()