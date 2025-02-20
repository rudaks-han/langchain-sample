import pymupdf

# file_path = "./sample/SPRI_AI_Brief_2023년12월호_F.pdf"
# file_path = "../sample/invoice.pdf"
# file_path = "./sample/invoice_sample.pdf"
# file_path = "./sample/table_sample.pdf"
file_path = "../sample/샘플.pdf"

doc = pymupdf.open(file_path)

for page in doc:  # iterate the document pages
    text = page.get_text()  # get plain text encoded as UTF-8
    print(text)
