from collections import OrderedDict

from img2table.document import PDF
from img2table.ocr import PaddleOCR
from pypdf import PdfReader

# img_path = "table.png"
# Definition of image from path
# img_from_path = Image(src=img_path)
# file_path = "../../sample2/BM202404290000031873_0.pdf"

# ocr = TesseractOCR(n_threads=1, lang="kor")
ocr = PaddleOCR(lang="korean", kw={"use_dilation": False, "use_angle_cls": True})
# ocr = EasyOCR(lang=["ko"], kw={"gpu": False})
# ocr = ocrmypdf

# file_path = "../../sample2/BM202405080000032619_0.pdf"
# file_path = "../../sample/table_sample.pdf"
# file_path = "../../sample2/BM202404290000031873_0.pdf"
file_path = "../../sample2/BM202404110000030261_0.pdf"
pdf = PDF(src=file_path)
extracted_tables = pdf.extract_tables(
    ocr=ocr,
    implicit_rows=True,
    implicit_columns=True,
    borderless_tables=True,
    min_confidence=50,
)

print(extracted_tables)

pdf = PdfReader(file_path)
for page in pdf.pages:
    print(f"mediabox: {page.mediabox}")


def to_markdown_table(ordered_dict: OrderedDict):
    markdown = []
    for i, row in enumerate(ordered_dict.values()):
        td_texts = []
        for table_cel in row:
            print(table_cel)
            if table_cel.value is None:
                table_cel.value = ""
            table_cel.value = table_cel.value.replace("\n", "<br>")
            td_texts.append(table_cel.value)

        markdown.append("| " + " | ".join(td_texts) + " |")

        if i == 0:  # 헤더 아래에 구분선 추가
            markdown.append("|" + " --- |" * len(td_texts))

    return "\n".join(markdown)

    # headers = ["Column 1", "Column 2", "Column 3", "Column 4"]
    # markdown = "| " + " | ".join(headers) + " |\n"
    # markdown += "| " + " | ".join(["-" * len(header) for header in headers]) + " |\n"

    # Data rows
    # for row in data.values():
    #     row_values = [cell["value"] for cell in row]
    #     markdown += "| " + " | ".join(row_values) + " |\n"
    #
    # return markdown


for page, tables in extracted_tables.items():
    for idx, table in enumerate(tables):
        table_cells: list[OrderedDict] = table.content
        # print(f"____ table bbox {table.bbox}")
        # for key, cells in table_cells.items():
        #     for cell in cells:
        #         print(f"[{cell.bbox}] {cell.value}")

        print(to_markdown_table(table_cells))
        # display_markdown(table.content, raw=True)
        # display_markdown(
        #     # table.html_repr(title=f"Page {page + 1} - Extracted table n°{idx + 1}"),
        #     table.content,
        #     raw=True,
        # )
        # display_html(
        #     table.html_repr(title=f"Page {page + 1} - Extracted table n°{idx + 1}"),
        #     raw=True,
        # )
