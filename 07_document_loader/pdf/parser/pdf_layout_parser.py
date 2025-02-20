import pdfplumber
from img2table.document import PDF
from langchain_core.documents import Document
from langchain_core.documents.base import Blob
from pdfplumber.page import Page

from pdf_layout_image_utils import extract_all_images, extract_valid_images
from pdf_layout_ocr_utils import to_markdown_tesseract, load_ocr
from pdf_layout_string_utils import (
    is_new_line_start,
    need_text_heading,
    is_new_line_detected,
)
from pdf_layout_utils import (
    get_all_chars,
    is_header_text,
    is_footer_text,
)
from table_parser_utils import (
    table_to_markdown,
    debug_tablefinder,
    extract_table_merged_cells,
    is_text_within_table,
    is_text_within_table_ocr,
)


class PdfLayoutParser:
    def __init__(self, file_path: str, **kwargs):
        self.file_path = file_path
        self.pdf = pdfplumber.open(file_path)
        self.ocr_extracted_tables = []
        self.pages = self.pdf.pages
        self.output_texts = []
        self.table_append_to_page = []
        self.image_append_to_page = []
        self.image_save_dir = kwargs.get("image_save_dir", "./images")
        self.line_height = kwargs.get("line_height", 16)
        self.debug = kwargs.get("debug", False)
        self.debug_table = kwargs.get("debug_table", False)
        self.extract_images = kwargs.get("extract_images", False)
        self.use_table_ocr = kwargs.get("use_table_ocr", False)
        self.center_margin_width = kwargs.get(
            "center_margin_width", 20
        )  # 가로로 나눠진 페이지의 중앙 여백 (가로 문서 중 반으로 나눠졌는지 여부 확인용)
        self.remove_header = kwargs.get("remove_header", True)
        self.remove_footer = kwargs.get("remove_footer", True)
        self.header_margin = kwargs.get("remove_header", 85)  # 헤더로 인지할 마진
        self.footer_margin = kwargs.get("footer_margin", 85)  # footer로 인지할 마진

    def execute(self):
        if self.use_table_ocr:
            self.execute_ocr()
        docs = self.load()

        return docs
        # return "".join(self.output_texts)

    def save_to_markdown(self, output, output_path: str):
        with open(output_path, "w") as f:
            f.write(output)

    def append_image(self, all_images, top):
        valid_images = extract_valid_images(all_images, top)

        if valid_images:
            for image in valid_images:
                self.append_text("\n\n")
                img_str = (
                    f"![{image["img_name"]}]({image["img_path"]}/{image['img_name']})"
                )
                self.append_text(img_str)
                self.append_text("\n\n")

    def append_text(self, text: str):
        self.output_texts.append(text)

    def print_page_info(self, page: Page):
        page_width = page.width
        page_height = page.height

        print(f"============== {page.page_number} ==============")
        print(f"page size: {page_width} x {page_height}")
        print("image count:", len(page.images))
        # print("table count:", len(page.find_tables(table_settings=table_settings)))
        print(f"================================")

    def execute_ocr(self):
        ocr = load_ocr()
        pdf = PDF(src=self.file_path)
        extracted_tables = pdf.extract_tables(
            ocr=ocr,
            implicit_rows=False,
            implicit_columns=False,
            borderless_tables=False,
            min_confidence=50,
        )
        if extracted_tables:
            scale = 2.77
            for key, value in extracted_tables.items():
                for table in value:
                    table_info = {
                        "page": key + 1,
                        "bbox": {
                            table.bbox.x1 / scale,
                            table.bbox.y1 / scale,
                            table.bbox.x2 / scale,
                            table.bbox.y2 / scale,
                        },
                        "markdown": to_markdown_tesseract(table.content),
                    }
                    self.ocr_extracted_tables.append(table_info)

    def load(self):
        blob = Blob.from_data(open(self.file_path, "rb").read(), path=self.file_path)
        return self.parse(blob)

        # for i, page in enumerate(self.pages):
        #     if is_horizontal_page(page):
        #         if is_horizontal_split_page(page, self.center_margin_width):
        #             left_page = page.within_bbox((0, 0, page.width / 2, page.height))
        #             right_page = page.within_bbox(
        #                 (page.width / 2, 0, page.width, page.height)
        #             )
        #             self.parse_page(left_page)
        #             self.parse_page(right_page)
        #         else:
        #             self.parse_page(page)
        #     else:
        #         self.parse_page(page)

    def parse(self, blob: Blob):
        with blob.as_bytes_io() as file_path:  # type: ignore[attr-defined]
            doc = pdfplumber.open(self.file_path)  # open document

            return [
                Document(
                    page_content=self.parse_page(page) + "\n",
                    metadata=dict(
                        {
                            "source": blob.source,  # type: ignore[attr-defined]
                            "file_path": blob.source,  # type: ignore[attr-defined]
                            "page": page.page_number - 1,
                            "total_pages": len(doc.pages),
                        },
                        **{
                            k: doc.metadata[k]
                            for k in doc.metadata
                            if type(doc.metadata[k]) in [str, int]
                        },
                    ),
                )
                for page in doc.pages
            ]

    def parse_page(self, page: Page):
        self.output_texts.clear()
        table_settings = {}

        if self.debug_table:
            debug_tablefinder(page, table_settings)

        find_tables = page.find_tables(table_settings=table_settings)
        extract_tables = page.extract_tables(table_settings=table_settings)
        ocr_extracted_tables = self.ocr_extracted_tables

        table_merged_cells = extract_table_merged_cells(page, table_settings)
        all_images = extract_all_images(
            page, self.image_append_to_page, self.image_save_dir
        )
        all_chars = get_all_chars(page)
        # all_words = get_all_words(page)

        prev_x0 = page.width
        prev_x1 = page.width
        prev_bottom = 0
        prev_height = 0

        if self.debug:
            self.print_page_info(page)

        for char_index, char in enumerate(all_chars):
            if self.remove_header and is_header_text(
                self.header_margin, char["bottom"]
            ):
                continue
            elif self.remove_footer and is_footer_text(
                page.height - self.footer_margin, char["top"]
            ):
                continue

            x0, top, x1, bottom, top, bottom, doctop, size, width, height = (
                char["x0"],
                char["top"],
                char["x1"],
                char["bottom"],
                char["top"],
                char["bottom"],
                char["doctop"],
                char.get("size", -1),
                char.get("width", -1),
                char.get("height", -1),
            )

            if self.extract_images:
                self.append_image(all_images, top)

            table_index, table = is_text_within_table(char, find_tables, extract_tables)
            table_index_ocr, table_ocr = is_text_within_table_ocr(
                char, ocr_extracted_tables
            )

            if table_ocr is not None:
                markdown = table_ocr["markdown"]
                table_uid = f"{page.page_number}_{markdown}"

                if table_uid not in self.table_append_to_page:
                    self.append_text("\n\n")
                    self.append_text(markdown)
                    self.append_text("\n\n")
                    self.table_append_to_page.append(table_uid)
            elif table is not None:
                table_uid = f"{page.page_number}_{" ".join(cell for row in table for cell in row if cell is not None)}"

                self.print_debug(char, prev_bottom, prev_height, prev_x0, None)

                if table_uid not in self.table_append_to_page:
                    markdown_table = table_to_markdown(
                        table, table_index, table_merged_cells
                    )
                    self.append_text(markdown_table)
                    self.table_append_to_page.append(table_uid)
            else:
                text = char["text"]
                if is_new_line_start(char, prev_x1):
                    self.append_text("\n")
                    first_char_of_line = True
                else:
                    first_char_of_line = False

                if is_new_line_detected(
                    top=top,
                    prev_bottom=prev_bottom,
                    prev_height=prev_height,
                    line_height=self.line_height,
                ):
                    self.append_text("\n")

                if not first_char_of_line:
                    self.append_text(" ")

                if need_text_heading(char, first_char_of_line):
                    self.append_text(f"# {text}")
                else:
                    self.append_text(text)

                self.print_debug(
                    char, prev_bottom, prev_height, prev_x0, first_char_of_line
                )

            prev_bottom = top + height
            prev_x0 = x0
            prev_x1 = x1
            prev_height = height

        return "".join(self.output_texts)

    def print_debug(self, char, prev_bottom, prev_height, prev_x0, first_char_of_line):
        if self.debug:
            text, x0, top, x1, bottom, top, bottom, doctop, size, width, height = (
                char["text"],
                char["x0"],
                char["top"],
                char["x1"],
                char["bottom"],
                char["top"],
                char["bottom"],
                char["doctop"],
                char.get("size", -1),
                char.get("width", -1),
                char.get("height", -1),
            )
            print(
                f" [{text}] {round(x0), round(top), round(x1), round(bottom)}, width: {round(width)}, height: {height}, size: {round(size)}, prev_bottom: {round(prev_bottom)}, prev_height: {prev_height}, spacing: {round(x0 - prev_x0)}, first_char_of_line: {first_char_of_line}",
            )
