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

import os, json, numpy as np, faiss
from sentence_transformers import sentence_transformers

#Specific lecture number
num = "01"
lecture_num = "lecture{num}_processed.json"

#Different paths
PROCESSED_PATH = "data/{lecture_num}"
INDEX_OUT = "data/index_{num}.faiss"
META_OUT = "data/meta_{lnum}.json"
MODEL_TYPE = "sentence-transformers/all-MiniLM-L6-v2"

#Load Sections
def creating_sections(processed_path):
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
            "doc" : lecture_num
            "page" : page_num
            "text" : out_of_context_text[:500]
        })

    #Check that the text is successfully loaded
    print(f"Loaded {len(full_text)} sections")

    return full_text, meta_data



#Embed - Start using references
def embed_text(full_text):
    #Loading the model and settings - specifically "used to map sentences/text to embeddings"
    current_model = SentenceTransformer(MODEL_TYPE)         #saved specifically locally
    
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
    #Find the number of columns needed for the indexing
    embeddings_dimension = embeddings.shape[1]

    #Reference 4 for lines 87-91
    #Using cosine similarity to see how "similiar/same direction"; unlike the example given in 4
    #index => "encapsulates the set of database vecotrs, and optionally preprocesses them to make searching efficient"
    index = faiss.IndexFlatIP(embeddings_dimension)
    #adding the ids to the indexed vectors
    index.add(embeddings)

    #Confirm that it did index
    print(f"Indexed {index.ntotal} vectors (dimension = [embeddings_dimension])")

    return index

#Main function and save file
def main():
    #global var that will become var in main
    text = []  #full text that gets referenced by meta
    data = []  #data that will get loaded

    text, data = creating_sections(PROCESSED_PATH)
    curr_embed = embed_text(text)
    curr_index = build_faiss_index(curr_embed)

    #Save to file
    faiss.write_index(curr_index, INDEX_OUT)
    with open(META_OUT, "w") as file:
        json.dump(meta, f, indent = 2)

    #Confirm the save
    print(f"Saved index -> {INDEX_OUT}")
    print(f"Saved meta  -> {META_OUT}")

    return

main()