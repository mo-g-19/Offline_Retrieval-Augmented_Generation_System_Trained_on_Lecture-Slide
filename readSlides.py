import fitz #pyMuPDF library
import json

input_doc = "06_Naming.pdf"
output_doc = "lecture06_processed.json"


doc = fitz.open(input_doc)

slides = []

for number, page in enumerate(doc):     #to go through in order of the slides
    raw_text = page.get_text()
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]     #creates a list of stripped line that doesn't have any empty lines
    indv_slide_txt = " ".join(lines)    #puts all the lines from one slide into one paragraph
    #Creating a dictionary inside the slides array 
    slides.append({
        "id": f"slide{number+1}",
        "text": indv_slide_txt
    })

with open(output_doc, "w") as file:
    json.dump(slides, file, indent = 2)

print(f"Successfully Extracted {len(slides)} slides to {output_doc}")
    #print(indv_slide_txt)
    #text = page.get_text("blocks")  #getting info on the text segments
    #print(text[2])
    #if (int(text[2]) < 700) :    #sorting out the footer
        #if (text[])
     #   print(text[5])

"""Current issues with this:
    1) Does not differenciate between Slide name and other text
    2) Is not able to tell that rows in a collumn are connected (Ex: Intro, last slide)
    3) Can't tell the difference in indentation in bullet notes
    4) Includes the symbol for the bulletnotes"""