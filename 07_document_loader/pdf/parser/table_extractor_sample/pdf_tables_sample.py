file_path = "../sample2/BM202404110000030261_0.pdf"
fileobj = open(file_path, "rb")

from pdftables.pdf_document import PDFDocument

doc = PDFDocument.from_fileobj(fileobj)

from pdftables.pdftables import page_to_tables

page = doc.get_page(pagenumber)
tables = page_to_tables(page)

from pdftables.pdftables import page_to_tables

for page_number, page in enumerate(doc.get_pages()):
    tables = page_to_tables(page)
