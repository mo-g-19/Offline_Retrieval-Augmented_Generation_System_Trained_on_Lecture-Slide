import fitz #pyMuPDF library
import json

doc = fitz.open("01_Introduction.pdf")

slides = []

for number, page in enumerate(doc):     #to go through in order of the slides
    raw_text = page.get_text()
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