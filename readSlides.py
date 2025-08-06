import fitz #pyMuPDF library

doc = fitz.open("01_Introduction.pdf")

for page in doc:
    text = page.get_text()
    print(text)