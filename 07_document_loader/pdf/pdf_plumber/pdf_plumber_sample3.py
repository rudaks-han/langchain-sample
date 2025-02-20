import os

import pdfplumber
from pdfplumber.table import Table

# file_path = "./sample/SPRI_AI_Brief_2023년12월호_F.pdf"
# file_path = "./sample/invoice_sample.pdf"
# file_path = "./sample/invoice.pdf"
# file_path = "../sample/샘플.pdf"
# file_path = "../sample/샘플2.pdf"
# file_path = "../sample/ag-energy-round-up-2017-02-24.pdf"
# file_path = "../sample/background-checks.pdf"
# file_path = "../sample/ca-warn-report.pdf"
# file_path = "../sample/san-jose-pd-firearm-sample.pdf"
# file_path = "../sample2/BM202403270000028996_0.pdf"
# file_path = "../sample/table_sample.pdf"
# file_path = "../sample/샘플.pdf"
file_path = "/Users/rudaks/temp/pdf/샘플.pdf"

debug = False

table_settings = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
}


def is_within_bounds(x0, y0, x1, y1, bound_x0, bound_y0, bound_x1, bound_y1):
    return (bound_x0 <= x0 <= bound_x1 and bound_y0 <= y0 <= bound_y1) and (
        bound_x0 <= x1 <= bound_x1 and bound_y0 <= y1 <= bound_y1
    )


def is_within_tables(find_tables, extract_tables, x0, y0, x1, y1) -> Table:
    for i, table in enumerate(find_tables):
        table_x0, table_y0, table_x1, table_y1 = table.bbox

        if is_within_bounds(
            x0,
            y0,
            x1,
            y1,
            table_x0,
            table_y0,
            table_x1,
            table_y1,
        ):
            return extract_tables[i]

    return None


def table_to_markdown(table):
    filtered_row = [value for value in table[0] if value is not None]
    filtered_row_len = len(filtered_row)

    if filtered_row_len > 1:
        result = "| " + " | ".join(filtered_row) + " |" + "\n"
        result += "| " + " | ".join(["---"] * len(filtered_row)) + " |\n"

        for row in table[1:]:
            column = []
            for col_index, col in enumerate(row):
                if col is not None and col != "":
                    column.append(col)

            if filtered_row_len == len(column):
                result += "| " + " | ".join(column) + " |\n"
    else:
        result = ""
        for row in table:
            if row[0] is not None and row[0] != "":
                result += row[0] + "\n"
    return result


def check_image_to_render(image_info, pos_y):
    render_images = []
    for i in range(len(image_info) - 1, -1, -1):
        if image_info[i]["pos_y"] < pos_y:
            render_images.append(image_info[i])
            del image_info[i]

    return render_images


pdf = pdfplumber.open(file_path)
pages = pdf.pages

result = []

table_append_to_page = []
image_append_to_page = []
all_images = []
image_output_dir = "./images"


def get_image_info(page, images):
    image_info = []

    for image in images:

        info = {
            "pos_x": image["x0"],
            "pos_y": image["top"],
            "width": image["width"],
            "height": image["height"],
        }
        image_info.append(info)

        x0, top, x1, bottom = image["x0"], image["top"], image["x1"], image["bottom"]
        # image_data = page.crop((x0, top, x1, bottom)).to_image().original
        image_data = page.crop((x0, top, x1, bottom)).to_image(resolution=100)
        img_save_path = os.path.join(
            image_output_dir, f"page_{page.page_number}_img_{page.page_number}.png"
        )
        image_data.save(img_save_path, format="PNG")

    # return result
    # return img_x0, image_top, img_x1, image_bottom
    return image_info


def parse_page(page_no, page):
    find_tables = page.find_tables(table_settings=table_settings)
    extract_tables = page.extract_tables(table_settings=table_settings)
    chars = page.chars
    images = page.images
    page_width = page.width
    page_height = page.height
    page_number = page.page_number

    page_x0, page_y0, page_x1, page_y1 = (
        page.bbox[0],
        page.bbox[1],
        page.bbox[2],
        page.bbox[3],
    )

    if debug:
        print(f"---------- {page_number} page -----------------")
        print(f"find_tables: {len(find_tables)}")
        print(f"page_number: {page_number}")
        print(f"page size: {page_width} x {page_height}")
        print(f"image size: {len(images)}")
        print(f"image info: {get_image_info(page, images)}")
        print("---------------------------------")

    elements = []

    image_info = get_image_info(page, images)

    # 텍스트 요소 추가
    for char in chars:
        elements.append(
            {
                "type": "text",
                "x0": char["x0"],
                "y0": char["y0"],
                "x1": char["x1"],
                "y1": char["y1"],
                "content": char,
            }
        )

    # 정렬된 요소 순서대로 출력
    for element in elements:
        if element["type"] == "text":
            char = element["content"]

            text_x0, text_y0, text_x1, text_y1 = (
                char["doctop"] - (page_no - 1) * page_y1,
                char["top"],
                char["doctop"] + char["height"] - (page_no - 1) * page_y1,
                char["bottom"],
            )
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

            pos_x = x0
            pos_y = top
            last_post_y = pos_y

            table = is_within_tables(
                find_tables,
                extract_tables,
                text_x0,
                text_y0,
                text_x1,
                text_y1,
            )
            if table is not None:
                markdown_table = table_to_markdown(table)
                if markdown_table not in table_append_to_page:
                    result.append("\n\n")
                    result.append(markdown_table)
                    result.append("\n\n")
                    table_append_to_page.append(markdown_table)
            else:
                text = char["text"]
                if debug:
                    text += (
                        f" [text] {round(x0), round(top), round(x1), round(bottom)}\n"
                    )

                result.append(text)

            render_images = check_image_to_render(image_info, last_post_y)
            if render_images:
                for image in render_images:
                    result.append("\n\n")
                    result.append("[이미지]")
                    if debug:
                        result.append(
                            f" x: {round(image['pos_x'])}, y: {round(image['pos_y'])}, w: {round(image['width'])}, h: {round(image['height'])}"
                        )
                    result.append("\n\n")


for page_no, page in enumerate(pages):
    parse_page(page_no + 1, page)

print("".join(result))

# parse_page(1, page=pages[0])
# parse_page(2, page=pages[1])
# print("".join(result))
