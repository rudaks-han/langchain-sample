import pandas as pd
import pdfplumber
from pdfplumber.utils import obj_to_bbox, get_bbox_overlap, extract_text


def process_pdf(pdf_path):
    pdf = pdfplumber.open(pdf_path)
    all_text = []

    for page in pdf.pages:
        filtered_page = page
        chars = filtered_page.chars

        for table in page.find_tables():
            first_table_char = page.crop(table.bbox).chars[0]
            filtered_page = filtered_page.filter(
                lambda obj: get_bbox_overlap(obj_to_bbox(obj), table.bbox) is None
            )
            chars = filtered_page.chars

            df = pd.DataFrame(table.extract())
            df.columns = df.iloc[0]
            markdown = df.drop(0).to_markdown(index=False)

            chars.append(first_table_char | {"text": markdown})

        page_text = extract_text(chars, layout=True)
        all_text.append(page_text)

    pdf.close()
    return "\n".join(all_text)


# file_path = "./sample/SPRI_AI_Brief_2023년12월호_F.pdf"
file_path = "../sample/샘플.pdf"
# file_path = "./sample/invoice_sample.pdf"
# file_path = "./sample/invoice.pdf"
# file_path = "../sample/table_sample.pdf"

extracted_text = process_pdf(file_path)
print(extracted_text)
