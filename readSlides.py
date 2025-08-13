"""The purpose of this document is to take the text from the pdf version of the slides and convert it into a dictionary where the key is the slide number and the value is the text"""

import fitz     #pyMuPDF library: to read text from a pdf (needed to install)
import json     #to save the individual presentation as a document with a dictionary that has a value for the slides and text in the slides
import argparse #Uses flags, so I don't have to hard code the presentation slides file path



doc = fitz.open(input_doc)

#The single array that holds the dictionaries
slides = []

main(): 
    #instead of hard coding the input (pdf slides) and output (json text file) path, make it part of the command prompt
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-pdf", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    #The pdf document to read from and the new document to save information
    input_doc = args.input_pdf
    output_doc = args.output_json

    #Going through each slide in the presentation
    for number, page in enumerate(doc):     #to go through in order of the slides
        raw_text = page.get_text()
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]     #creates a list of stripped line that doesn't have any empty lines
        indv_slide_txt = " ".join(lines)    #puts all the lines from one slide into one paragraph
        #Creating a dictionary inside the slides array 
        slides.append({
            "id": f"slide{number+1}",
            "text": indv_slide_txt
        })

    #Saves all the information in the ouput_doc
    with open(output_doc, "w") as file:
        json.dump(slides, file, indent = 2)

    print(f"Successfully Extracted {len(slides)} slides to {output_doc}")

main()