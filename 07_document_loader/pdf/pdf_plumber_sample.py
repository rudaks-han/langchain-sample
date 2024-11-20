import pdfplumber

file_path = "./sample/SPRI_AI_Brief_2023년12월호_F.pdf"


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"[doc {i+1}]" + d.page_content for i, d in enumerate(docs)]
        )
    )


if __name__ == "__main__":
    pdf = pdfplumber.open(file_path)
    pages = pdf.pages
    print("총 페이지 수: ", len(pages))
    for i, page in enumerate(pages):
        print(f"[text] {page.extract_text()}")
        print(f"[table] {page.extract_table()}")

        page_height = page.height
        img = page.images[0]
        if img is not None:
            boxpoint = (
                img["x0"],
                page_height - img["y1"],
                img["x1"],
                page_height - img["y0"],
            )
            copy_crop = page.crop(boxpoint)
            image_object = copy_crop.to_image(resolution=400)
