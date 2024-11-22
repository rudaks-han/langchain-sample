import io
import os
import pathlib

import PIL
import fitz
import pandas as pd
import pymupdf4llm

file_path = "./sample/invoice_sample.pdf"
file_path = "./sample/샘플.pdf"

doc = fitz.open(file_path)  # open example file


def show_image(item, title=""):
    """Display a pixmap.

    Just to display Pixmap image of "item" - ignore the man behind the curtain.

    Args:
        item: any PyMuPDF object having a "get_pixmap" method.
        title: a string to be used as image title

    Generates an RGB Pixmap from item using a constant DPI and using matplotlib
    to show it inline of the notebook.
    """
    DPI = 150  # use this resolution
    import numpy as np
    import matplotlib.pyplot as plt

    # %matplotlib inline
    pix = item.get_pixmap(dpi=DPI)
    img = np.ndarray([pix.h, pix.w, 3], dtype=np.uint8, buffer=pix.samples_mv)
    plt.figure(dpi=DPI)  # set the figure's DPI
    plt.title(title)  # set title of image
    _ = plt.imshow(img, extent=(0, pix.w * 72 / DPI, pix.h * 72 / DPI, 0))


def save_as_markdown():
    md_text = pymupdf4llm.to_markdown(file_path)
    pathlib.Path("output.md").write_bytes(md_text.encode())


# https://github.com/pymupdf/PyMuPDF-Utilities/blob/master/table-analysis/join_tables.ipynb
def to_table():
    page = doc[0]  # read first page to demo the layout
    show_image(page, "First Page Content")

    dataframes = []  # list of DataFrames per table fragment

    for page in doc:  # iterate over the pages
        tabs = page.find_tables()  # locate tables on page
        if len(tabs.tables) == []:  # no tables found?
            break  # stop
        tab = tabs[0]  # assume fragment to be 1st table
        dataframes.append(tab.to_pandas())  # append this DataFrame

    df = pd.concat(dataframes)  # make concatenated DataFrame
    print(df)


def extract_images(pdf: fitz.Document, page: int, imgDir: str = "img"):
    imageList = pdf[page].get_images()
    os.makedirs(imgDir, exist_ok=True)
    if imageList:
        print(page)
        for idx, img in enumerate(imageList, start=1):
            data = pdf.extract_image(img[0])
            with PIL.Image.open(io.BytesIO(data.get("image"))) as image:
                image.save(f'{imgDir}/{page}-{idx}.{data.get("ext")}', mode="wb")


if __name__ == "__main__":
    # save_as_markdown()
    # to_table()
    # to_image()
    for page in range(doc.page_count):
        extract_images(doc, page)
