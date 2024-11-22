import openparse
from openparse import TextElement, TableElement
from openparse.schemas import ImageElement

file_path = "./sample/invoice_sample.pdf"
# file_path = "./sample/샘플.pdf"
# file_path = "./sample/SPRI_AI_Brief_2023년12월호_F.pdf"
# file_path = "./sample/table_sample.pdf"
# file_path = "./sample/111.pdf"
# file_path = "./sample/invoice.pdf"


parser = openparse.DocumentParser(
    table_args={"parsing_algorithm": "pymupdf", "table_output_format": "markdown"}
)
doc = parser.parse(file_path, ocr=False)

# print(doc)
for node in doc.nodes:
    # print(node)
    for element in node.elements:
        if isinstance(element, TextElement):
            print(f"[text]")
            # print(f"[text] {element.embed_text}")
        elif isinstance(element, ImageElement):
            print("[image]")
        elif isinstance(element, TableElement):
            print("table")
        else:
            print("else")
