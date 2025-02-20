import inspect
from math import floor

from pdfplumber.page import Page
from pdfplumber.utils.text import WordExtractor


def is_horizontal_page(page: Page):
    return page.width > page.height


# 가로로 나눠진 페이지는 가운데 라인에 글자가 있는지 여부로 판단한다.
def is_horizontal_split_page(page: Page, center_margin_width: int):
    center_line_page = page.within_bbox(
        (
            page.width / 2 - center_margin_width,
            0,
            page.width / 2 + center_margin_width,
            page.height,
        )
    )
    return center_line_page.extract_text() == ""


def is_header_text(header_margin, bottom):
    return bottom < header_margin


def is_footer_text(footer_margin, bottom):
    return bottom > footer_margin


def get_wordmap(page: Page):
    WORD_EXTRACTOR_KWARGS = inspect.signature(WordExtractor).parameters.keys()

    kwargs = {
        "layout_bbox": (0, 0, page.width, page.height),
        "layout_height": page.height,
        "layout_width": page.width,
        "presorted": True,
        "extra_attrs": ["fontname", "size"],
    }

    extractor = WordExtractor(
        **{k: kwargs[k] for k in WORD_EXTRACTOR_KWARGS if k in kwargs}
    )

    return extractor.extract_wordmap(page.chars)


def get_all_chars(page: Page):
    # wordmap = get_wordmap(page)
    # tuples = list(wordmap.tuples)
    # items = list(wordmap.tuples)
    # sorted_items = sorted(items, key=lambda x: x[0]["top"])
    # for item in tuples:
    #     print(item[1])

    all_chars = []
    items = page.extract_words()
    for item in items:
        all_chars.append(item)

    # items = list(wordmap.tuples)
    # sorted_items = sorted(items, key=lambda x: x[0]["top"])
    # items = sorted(items, key=custom_sort_key)
    # for item in items:
    #     word = item[0]
    #     for value in item[1]:
    #         bold = False
    #         if is_bold_fontname(word.get("fontname", "")):
    #             bold = True
    #         value["bold"] = bold
    #         all_chars.append(value)

    # print("--------- extract words ---------")
    # for word in page.extract_words():
    #     print(word["text"], end="")
    # print("--------- all chars ---------")
    # for item in items:
    #     for value in item[1]:
    #         print(value["text"], end="")
    # print("-----------------------------")

    return all_chars


def custom_sort_key(item):
    TOLERANCE = 20
    top = floor(item[0]["top"] / TOLERANCE) * TOLERANCE
    x0 = floor(item[0]["x0"] / TOLERANCE) * TOLERANCE
    return top, x0


def get_all_words(page: Page):
    wordmap = get_wordmap(page)

    all_words = []
    items = list(wordmap.tuples)
    sorted_items = sorted(items, key=custom_sort_key)

    for item in sorted_items:
        word = item[0]
        all_words.append(word)

    return all_words


def save_to_image(pdf_image, file_path):
    pdf_image.original.save(file_path)
