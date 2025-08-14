"""The purpose of this document is to take the text from the pdf version of the slides and convert it into a dictionary where the key is the slide number and the value is the text"""

import os
import fitz     #pyMuPDF library: to read text from a pdf (needed to install)
import json     #to save the individual presentation as a document with a dictionary that has a value for the slides and text in the slides
import argparse #Uses flags, so I don't have to hard code the presentation slides file path


def main(): 
    #The single array that holds the dictionaries
    slides = []

    #instead of hard coding the input (pdf slides) and output (json text file) path, make it part of the command prompt
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-pdf", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    #The pdf document to read from and the new document to save information
    input_pdf = os.path.abspath(args.input_pdf)
    output_doc = os.path.abspath(args.output_json)
    os.makedirs(os.path.dirname(output_doc), exist_ok=True)

    #Going through each slide in the presentation
    with fitz.open(input_pdf) as doc:   #to go through in order of the slides
        for raw_slides in range (doc.page_count):
            pg = doc.load_page(raw_slides)
            text = pg.get_text("text") or ""
            slides.append({
                "doc": os.path.basename(output_doc),
                "id": f"slide{raw_slides+1}",
                "text": text
            })

    #Saves all the information in the ouput_doc
    with open(output_doc, "w", encoding="utf-8") as file:
        json.dump(slides, file, ensure_ascii=False, indent = 2)

    print(f"Successfully Extracted {len(slides)} slides to {output_doc}")

main()