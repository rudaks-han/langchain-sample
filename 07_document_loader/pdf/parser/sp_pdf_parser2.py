import inspect
import os
import pdfplumber
from math import floor
from pdfplumber.page import Page
from pdfplumber.utils.text import WordExtractor

from pdf_layout_image_utils import extract_valid_images
from pdf_layout_string_utils import is_bold_fontname
from table_parser_utils import (
    table_to_markdown,
    debug_tablefinder,
    table_row_col_mark,
    check_merged_cells,
    is_within_tables,
)

table_settings = {
    # "vertical_strategy": "lines",
    # "horizontal_strategy": "lines",
    # "snap_tolerance": 0,  # 가까운 텍스트 병합 (필요 시 조정)
    # "join_tolerance": 0,  # 인접 텍스트 병합 오차 (필요 시 조정)
    # "intersection_tolerance": 0,  # 교차점 감지 오차 (필요 시 조정)
}


# table_settings = {
#     "vertical_strategy": "lines_strict",
#     "horizontal_strategy": "lines_strict",
#     "explicit_vertical_lines": [],
#     "explicit_horizontal_lines": [],
#     "snap_tolerance": 3,
#     "snap_x_tolerance": 3,
#     "snap_y_tolerance": 3,
#     "join_tolerance": 3,
#     "join_x_tolerance": 3,
#     "join_y_tolerance": 3,
#     "edge_min_length": 3,
#     "min_words_vertical": 3,
#     "min_words_horizontal": 1,
#     "intersection_tolerance": 3,
#     "intersection_x_tolerance": 3,
#     "intersection_y_tolerance": 3,
#     "text_tolerance": 3,
#     "text_x_tolerance": 3,
#     "text_y_tolerance": 3,
# }


class SpPdfParser:
    def __init__(self, file_path: str, debug: bool = False):
        self.WORD_EXTRACTOR_KWARGS = inspect.signature(WordExtractor).parameters.keys()
        self.file_path = file_path
        self.pdf = pdfplumber.open(file_path)
        self.pages = self.pdf.pages
        self.debug = debug
        self.output = []
        self.table_append_to_page = []
        self.image_append_to_page = []
        self.image_output_dir = "./images"
        self.line_height = 30
        self.letter_spacing = 18
        self.debug_table = True

    def execute(self):
        self.initialize()
        self.parse()

        return "".join(self.output)

    def get_all_chars(self, page: Page):
        kwargs = {
            "layout_bbox": (0, 0, page.width, page.height),
            "layout_height": page.height,
            "layout_width": page.width,
            "presorted": True,
            "extra_attrs": ["fontname", "size"],
        }

        extractor = WordExtractor(
            **{k: kwargs[k] for k in self.WORD_EXTRACTOR_KWARGS if k in kwargs}
        )
        wordmap = extractor.extract_wordmap(page.chars)

        all_chars = []
        items = list(wordmap.tuples)
        sorted_items = sorted(items, key=lambda x: x[0]["top"])

        for item in sorted_items:
            word = item[0]
            for value in item[1]:
                bold = False
                if is_bold_fontname(word.get("fontname", "")):
                    bold = True
                value["bold"] = bold
                all_chars.append(value)

        return all_chars

    def custom_sort_key(self, item):
        TOLERANCE = 20
        top = floor(item[0]["top"] / TOLERANCE) * TOLERANCE
        x0 = floor(item[0]["x0"] / TOLERANCE) * TOLERANCE
        return top, x0

    def get_all_words(self, page: Page):
        kwargs = {
            "layout_bbox": (0, 0, page.width, page.height),
            "layout_height": page.height,
            "layout_width": page.width,
            "presorted": True,
            "extra_attrs": ["fontname", "size"],
        }

        extractor = WordExtractor(
            **{k: kwargs[k] for k in self.WORD_EXTRACTOR_KWARGS if k in kwargs}
        )
        wordmap = extractor.extract_wordmap(page.dedupe_chars().chars)

        all_words = []
        items = list(wordmap.tuples)
        sorted_items = sorted(items, key=self.custom_sort_key)

        for item in sorted_items:
            word = item[0]
            all_words.append(word)

        return all_words

    def save_to_markdown(self, output, output_path: str):
        with open(output_path, "w") as f:
            f.write(output)

    def initialize(self):
        pass

    def is_text_within_table(
        self,
        page: Page,
        char: dict,
        find_tables,
        extract_tables,
    ):
        text_x0, text_y0, text_x1, text_y1 = (
            char["doctop"] - (page.page_number - 1) * page.height,
            char["top"],
            char["doctop"] + char["height"] - (page.page_number - 1) * page.height,
            char["bottom"],
        )

        x0, top, x1, bottom = (
            char["x0"],
            char["top"],
            char["x1"],
            char["bottom"],
        )

        return is_within_tables(
            find_tables,
            extract_tables,
            x0,
            top,
            x1,
            bottom,
        )

    def append_image(self, all_images, top):
        render_images = extract_valid_images(all_images, top)
        if render_images:
            for image in render_images:
                self.append_to_output("\n\n")
                img_str = (
                    f"![{image["img_name"]}]({image["img_path"]}/{image['img_name']})"
                )
                self.append_to_output(img_str)
                if self.debug:
                    self.append_to_output(
                        f" x: {round(image['pos_x'])}, y: {round(image['pos_y'])}, w: {round(image['width'])}, h: {round(image['height'])}"
                    )
                self.append_to_output("\n\n")

    def append_to_output(self, text: str):
        self.output.append(text)

    def need_new_line(self, char, last_pos_x):
        if char["x0"] < last_pos_x:
            return True

        return False

    def need_char_spacing(self, char, last_pos_x, last_width):
        letter_spacing = 2
        if char["x0"] - (last_pos_x + last_width) - letter_spacing > 0:
            return True
        return False

    def need_text_heading(self, char, first_char_of_line):
        # if "□" in char["text"]:
        #     pass
        if (
            first_char_of_line
            # and char.get("size", -1) >= 16
            and is_bold_fontname(char.get("fontname", ""))
        ):
            return True
        return False

    def print_page_info(self, page: Page):
        page_width = page.width
        page_height = page.height

        print(f"============== {page.page_number} ==============")
        print(f"page size: {page_width} x {page_height}")
        print("image count:", len(page.images))
        # print("table count:", len(page.find_tables(table_settings=table_settings)))
        print(f"================================")

    def parse(self):
        for i, page in enumerate(self.pages):
            self.parse_page(page)

    def extract_all_images(self, page: Page, images):
        all_images = []

        for image_idx, image in enumerate(images):
            image_attrs = {
                "pos_x": image["x0"],
                "pos_y": image["top"],
                "width": image["width"],
                "height": image["height"],
            }

            x0, top, x1, bottom = (
                image["x0"],
                image["top"],
                image["x1"],
                image["bottom"],
            )
            # if self.debug:
            #     print(f"[check image] x0: {x0}, top: {top}, x1: {x1}, bottom: {bottom}")
            #     print(
            #         f"[check image] page_width: {page.width}, page.height: {page.height}"
            #     )

            background_image_margin = 0.1

            if (
                x1 + background_image_margin >= page.width
                or bottom + background_image_margin >= page.height
            ):
                print("[image] 페이지 범위 벗어남")
                continue
            else:
                # if self.debug:

                image_uid = f"{page.page_number}_{x0}_{top}_{x1}_{bottom}"
                if image_uid not in self.image_append_to_page:
                    # if self.debug:
                    #     print(
                    #         f"[image] x: {round(x0)}, y: {round(top)}, w: {round(x1)}, h: {round(bottom)}"
                    #     )

                    image_data = page.crop((x0, top, x1, bottom)).to_image(
                        width=image["width"]
                    )
                    filename = f"page_{page.page_number}_img_{image_idx}.png"
                    img_save_path = os.path.join(
                        self.image_output_dir,
                        filename,
                    )
                    image_data.save(img_save_path, format="PNG")
                    image_attrs["img_path"] = self.image_output_dir
                    image_attrs["img_name"] = filename
                    all_images.append(image_attrs)

                    self.image_append_to_page.append(image_uid)

        return all_images

    def curves_to_edges(self, cs):
        edges = []
        for c in cs:
            edges += pdfplumber.utils.rect_to_edges(c)
        return edges

    def parse_page(self, page: Page):
        # table_settings = {
        #     "vertical_strategy": "explicit",
        #     "horizontal_strategy": "explicit",
        #     "explicit_vertical_lines": self.curves_to_edges(page.curves + page.edges),
        #     "explicit_horizontal_lines": self.curves_to_edges(page.curves + page.edges),
        #     # "explicit_vertical_lines": self.curves_to_edges(page.curves) + page.edges,
        #     # "explicit_horizontal_lines": self.curves_to_edges(page.curves) + page.edges,
        #     "intersection_y_tolerance": 10,
        #     "snap_tolerance": 4,
        # }
        # table_settings = {
        #     "vertical_strategy": "lines",
        #     "horizontal_strategy": "text",
        #     "snap_y_tolerance": 5,
        #     "intersection_x_tolerance": 15,
        # }

        debug_tablefinder(page, table_settings)

        if page.page_number == 2:
            pass
        find_tables = page.find_tables(table_settings=table_settings)
        extract_tables = page.extract_tables(table_settings=table_settings)
        chars = page.chars
        images = page.images

        table_merged_cells = []
        for index, table in enumerate(page.find_tables(table_settings=table_settings)):
            row_col_mark = table_row_col_mark(table)
            for row_index, row in enumerate(table.rows):
                for col_index, cell in enumerate(row.cells):
                    if cell is None:
                        continue
                    merged = check_merged_cells(cell, row_col_mark)
                    if merged:
                        # table_merged_cells.append(f"{index}_{row_index}_{col_index}")
                        merged_info = {"table_index": index, "merged": merged}
                        # table_merged_cells.append(f"{index}_{row_index}_{col_index}")
                        table_merged_cells.append(merged_info)
                    else:
                        pass

        # all_chars = self.get_all_chars(page)
        all_words = self.get_all_words(page)

        last_pos_x = page.width
        last_pos_y = 0
        last_width = 0
        first_char_of_line = True

        # if self.debug:
        #     self.print_page_info(page)

        all_images = self.extract_all_images(page, images)

        for word in all_words:
            # x0, y0, x1, y1 = get_text_pos(word)
            x0, y0, x1, y1 = (
                word["x0"],
                word["top"],
                word["x1"],
                word["bottom"],
            )

            top, bottom, doctop = (
                word["top"],
                word["bottom"],
                word["doctop"],
            )
            size, width = (word.get("size", -1), word.get("width", -1))

            self.append_image(all_images, top)

            table_index, table = self.is_text_within_table(
                page, word, find_tables, extract_tables
            )

            if table is not None:
                table_uid = f"{page.page_number}_{" ".join(cell for row in table for cell in row if cell is not None)}"

                if table_uid not in self.table_append_to_page:
                    markdown_table = table_to_markdown(
                        table, table_index, table_merged_cells
                    )

                    self.append_to_output("\n\n")
                    self.append_to_output(markdown_table)
                    self.append_to_output("\n\n")
                    self.table_append_to_page.append(table_uid)
            else:
                text = word["text"]
                if self.need_new_line(word, last_pos_x):
                    self.append_to_output("\n")
                    first_char_of_line = True
                else:
                    first_char_of_line = False

                if top - last_pos_y > self.line_height:
                    self.append_to_output("\n")
                # if x0 - last_pos_x > self.letter_spacing:
                #     self.append_to_output(" ")
                if self.need_char_spacing(word, last_pos_x, last_width):
                    self.append_to_output(" ")

                if self.debug:
                    text += f" [text] {round(x0), round(top), round(x1), round(bottom)}, width: {width}, size: {size}, spacing: {x0-last_pos_x}, first_char_of_line: {first_char_of_line}\n"

                if self.need_text_heading(word, first_char_of_line):
                    self.append_to_output(f"## {text}")
                else:
                    self.append_to_output(text)

            last_pos_y = top
            last_pos_x = x0
            last_width = width

        # for char in all_chars:
        #     x0, y0, x1, y1 = get_text_pos(char)
        #
        #     top, bottom, doctop = (
        #         char["top"],
        #         char["bottom"],
        #         char["doctop"],
        #     )
        #     size, width = (char["size"], char["width"])
        #
        #     self.append_image(all_images, top)
        #
        #     table_index, table = self.is_text_within_table(
        #         page, char, find_tables, extract_tables
        #     )
        #
        #     if table is not None:
        #         table_uid = f"{page.page_number}_{" ".join(cell for row in table for cell in row if cell is not None)}"
        #
        #         if table_uid not in self.table_append_to_page:
        #             markdown_table = table_to_markdown(
        #                 table, table_index, table_merged_cells
        #             )
        #
        #             self.append_to_output("\n\n")
        #             self.append_to_output(markdown_table)
        #             self.append_to_output("\n\n")
        #             self.table_append_to_page.append(table_uid)
        #     else:
        #         text = char["text"]
        #         if self.need_new_line(char, last_pos_x):
        #             self.append_to_output("\n")
        #             first_char_of_line = True
        #         else:
        #             first_char_of_line = False
        #
        #         if top - last_pos_y > self.line_height:
        #             self.append_to_output("\n")
        #         # if x0 - last_pos_x > self.letter_spacing:
        #         #     self.append_to_output(" ")
        #         if self.need_char_spacing(char, last_pos_x, last_width):
        #             self.append_to_output(" ")
        #
        #         if self.debug:
        #             text += f" [text] {round(x0), round(top), round(x1), round(bottom)}, width: {width}, size: {size}, spacing: {x0-last_pos_x}, first_char_of_line: {first_char_of_line}\n"
        #
        #         if self.need_text_heading(char, first_char_of_line):
        #             self.append_to_output(f"## {text}")
        #         else:
        #             self.append_to_output(text)
        #
        #     last_pos_y = top
        #     last_pos_x = x0
        #     last_width = width
