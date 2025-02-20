import inspect
import os
import pdfplumber
from parser_utils import (
    is_within_tables,
    table_to_markdown,
    check_image_to_render,
)
from pdfplumber.page import Page
from pdfplumber.utils.text import WordExtractor

table_settings = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
}

WORD_EXTRACTOR_KWARGS = inspect.signature(WordExtractor).parameters.keys()


class SpPdfParser:
    def __init__(self, file_path: str, debug: bool = False):
        self.file_path = file_path
        self.pdf = pdfplumber.open(file_path)
        self.pages = self.pdf.pages
        self.debug = debug
        self.output = []
        self.table_append_to_page = []
        self.image_output_dir = "./images"
        self.line_height = 30
        self.letter_spacing = 20

    def execute(self):
        self.initialize()
        self.parse()

        return "".join(self.output)

    def save_to_markdown(self, output, output_path: str):
        with open(output_path, "w") as f:
            f.write(output)

    def initialize(self):
        pass

    def is_text_within_table(
        self,
        page: Page,
        char: str,
        find_tables,
        extract_tables,
    ):
        text_x0, text_y0, text_x1, text_y1 = (
            char["doctop"] - (page.page_number - 1) * page.height,
            char["top"],
            char["doctop"] + char["height"] - (page.page_number - 1) * page.height,
            char["bottom"],
        )

        table = is_within_tables(
            find_tables,
            extract_tables,
            text_x0,
            text_y0,
            text_x1,
            text_y1,
        )

        return table

    def append_to_output(self, text: str):
        self.output.append(text)

    def print_page_info(self, page: Page):
        page_width = page.width
        page_height = page.height

        print(f"============== {page.page_number} ==============")
        print(f"page size: {page_width} x {page_height}")
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
            image_data = page.crop((x0, top, x1, bottom)).to_image(width=image["width"])
            filename = f"page_{page.page_number}_img_{image_idx}.png"
            img_save_path = os.path.join(
                self.image_output_dir,
                filename,
            )
            image_data.save(img_save_path, format="PNG")
            image_attrs["img_path"] = self.image_output_dir
            image_attrs["img_name"] = filename
            all_images.append(image_attrs)

        return all_images

    def parse_page(self, page: Page):
        find_tables = page.find_tables(table_settings=table_settings)
        extract_tables = page.extract_tables(table_settings=table_settings)
        chars = page.chars
        images = page.images
        print("--------------------")
        print(page.extract_text())

        kwargs = {
            "layout_bbox": (0, 0, 595, 841),
            "layout_height": 841,
            "layout_width": 595,
            "presorted": True,
        }

        extractor = WordExtractor(
            **{k: kwargs[k] for k in WORD_EXTRACTOR_KWARGS if k in kwargs}
        )
        wordmap = extractor.extract_wordmap(chars)
        # kwargs = {}
        # kwargs.update(
        #     {
        #         "presorted": True,
        #         "layout_bbox": kwargs.get("layout_bbox") or objects_to_bbox(chars),
        #     }
        # )
        #
        # extractor = WordExtractor(
        #     **{k: kwargs[k] for k in WORD_EXTRACTOR_KWARGS if k in kwargs}
        # )
        # wordmap = extractor.extract_wordmap(chars)

        print("--------------------")
        last_pos_x = page.width
        last_pos_y = 0

        if self.debug:
            self.print_page_info(page)

        all_images = self.extract_all_images(page, images)

        for char in chars:
            x0, y0, x1, y1 = (
                char["x0"],
                char["y0"],
                char["x1"],
                char["y1"],
            )

            top, bottom, doctop = (
                char["top"],
                char["bottom"],
                char["doctop"],
            )

            table = self.is_text_within_table(page, char, find_tables, extract_tables)
            if table is not None:
                markdown_table = table_to_markdown(table)
                if markdown_table not in self.table_append_to_page:
                    self.append_to_output("\n\n")
                    self.append_to_output(markdown_table)
                    self.append_to_output("\n\n")
                    self.table_append_to_page.append(markdown_table)
            else:
                text = char["text"]
                if self.debug:
                    text += (
                        f" [text] {round(x0), round(top), round(x1), round(bottom)}\n"
                    )

                if x0 < last_pos_x:
                    self.append_to_output("\n")
                if top - last_pos_y > self.line_height:
                    self.append_to_output("\n")
                if x0 - last_pos_x > self.letter_spacing:
                    self.append_to_output(" ")

                self.append_to_output(text)

            last_pos_y = top
            last_pos_x = x0

            render_images = check_image_to_render(all_images, last_pos_y)
            if render_images:
                for image in render_images:
                    self.append_to_output("\n\n")
                    img_str = f"![{image["img_name"]}]({image["img_path"]}/{image['img_name']})"
                    self.append_to_output(img_str)
                    if self.debug:
                        self.append_to_output(
                            f" x: {round(image['pos_x'])}, y: {round(image['pos_y'])}, w: {round(image['width'])}, h: {round(image['height'])}"
                        )
                    self.append_to_output("\n\n")
