from collections import OrderedDict

from img2table.ocr import TesseractOCR


def load_ocr():
    # ocr = EasyOCR(lang=["ko"], kw={"gpu": False})
    # ocr = PaddleOCR(lang="korean", kw={"use_dilation": True, "use_angle_cls": True})
    # ocr = ocrmypdf

    return TesseractOCR(n_threads=1, lang="kor+eng")


def to_markdown_tesseract(ordered_dict: OrderedDict):
    markdown = []
    for i, row in enumerate(ordered_dict.values()):
        td_texts = []
        for table_cel in row:
            if table_cel.value is None:
                table_cel.value = ""

            value = table_cel.value.replace("\n", "<br>")
            td_texts.append(value)

        markdown.append("| " + " | ".join(td_texts) + " |")

        if i == 0:  # 헤더 아래에 구분선 추가
            markdown.append("|" + " --- |" * len(td_texts))

    return "\n".join(markdown)
