import pdfplumber

# file_path = "./sample/SPRI_AI_Brief_2023년12월호_F.pdf"
file_path = "./sample/샘플.pdf"
# file_path = "./sample/invoice_sample.pdf"
# file_path = "./sample/invoice.pdf"
# file_path = "./sample/table_sample.pdf"


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"[doc {i+1}]" + d.page_content for i, d in enumerate(docs)]
        )
    )


def not_within_bboxes(obj):
    """Check if the object is in any of the table's bbox."""

    def obj_in_bbox(_bbox):
        """See https://github.com/jsvine/pdfplumber/blob/stable/pdfplumber/table.py#L404"""
        v_mid = (obj["top"] + obj["bottom"]) / 2
        h_mid = (obj["x0"] + obj["x1"]) / 2
        x0, top, x1, bottom = _bbox
        return (h_mid >= x0) and (h_mid < x1) and (v_mid >= top) and (v_mid < bottom)

    return not any(obj_in_bbox(__bbox) for __bbox in bboxes)


if __name__ == "__main__":
    pdf = pdfplumber.open(file_path)
    pages = pdf.pages
    print("총 페이지 수: ", len(pages))
    for i, page in enumerate(pages):

        ts = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
        }

        # Get the bounding boxes of the tables on the page.
        bboxes = [table.bbox for table in page.find_tables(table_settings=ts)]

        # print("Text outside the tables:")
        # print(page.filter(not_within_bboxes).extract_text())
        # print(f"[tables] {page.extract_tables()}")

        print(f"[text] {page.extract_text()}")
        print(f"[tables] {page.extract_tables()}")

        lines = page.extract_text_lines()
        for line in lines:
            print(line)
        pass
        # page_height = page.height
        # img = page.images and page.images[0]
        # if len(img) > 0:
        #     x0 = img["x0"]
        #     x1 = img["x1"]
        #     y0 = page_height - img["y1"]
        #     y1 = page_height - img["y0"]
        #     if y1 > page_height:
        #         y1 = page_height
        #     boxpoint = (x0, y0, x1, y1)
        #     copy_crop = page.crop(boxpoint)
        #     image_object = copy_crop.to_image(resolution=400)
