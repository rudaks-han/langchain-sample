import pathlib

import pymupdf
import pymupdf4llm

# file_path = "./sample/SPRI_AI_Brief_2023년12월호_F.pdf"
# file_path = "../sample/invoice.pdf"
# file_path = "./sample/invoice_sample.pdf"
file_path = "./sample/table_sample.pdf"
# file_path = "./sample/샘플.pdf"

doc = pymupdf.open(file_path)


def to_markdown():
    # pymupdf4llm.LlamaMarkdownReader()
    md_text = pymupdf4llm.to_markdown(
        file_path,
        # hdr_info=True,
        embed_images=False,
        # image_size_limit=0,
        # write_images=True,
        image_path="./pymupdf/output/img",
        page_chunks=False,
        table_strategy="lines_strict",
        # table_strategy="lines",
    )
    import pathlib

    output_dir = pathlib.Path("pymupdf/output")
    output_file = output_dir / (pathlib.Path(file_path).stem + ".md")

    print(md_text)
    output_file.write_bytes(md_text.encode())


def to_file():
    with pymupdf.open(file_path) as doc:  # open document
        text = chr(12).join([page.get_text() for page in doc])
    pathlib.Path(file_path + ".txt").write_bytes(text.encode())


def to_data():
    for page in doc:  # iterate the document pages
        text = page.get_text().encode("utf8")  # get plain text (is in UTF-8)
        print(text)


def find_tables():
    for page in doc:  # iterate the document pages
        tabs = page.find_tables()  # locate and extract any tables on page
        print(f"{len(tabs.tables)} found on {page}")  # display number of found tables


if __name__ == "__main__":
    to_markdown()
    # to_file()
    # to_data()
    # find_tables()

# for page in doc:
#     text = page.get_text()
#     print(text)
