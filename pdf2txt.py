import PyPDF2

file = open("/home/dimus/dl/christophers_1960.pdf", "rb")

reader = PyPDF2.PdfReader(file)

txt = ''

for page in reader.pages:
    page_text = page.extract_text()
    txt += page_text + "\n"

print(txt)
