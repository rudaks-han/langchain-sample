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

debug = True

table_settings = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
}
# table_settings = {
#     # "vertical_strategy": "lines",
#     # "horizontal_strategy": "text",
#     "vertical_strategy": "lines",
#     "horizontal_strategy": "lines",
#     "snap_x_tolerance": 0,
#     "snap_y_tolerance": 0,
#     "intersection_x_tolerance": 0,
#     "intersection_y_tolerance": 0,
# }
# 잘되는것
# table_settings = {
#     "vertical_strategy": "lines",
#     "horizontal_strategy": "text",
#     "snap_x_tolerance": 5,
#     "snap_y_tolerance": 5,
#     "intersection_x_tolerance": 15,
# }


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


pdf = pdfplumber.open(file_path)
pages = pdf.pages

result = []

table_append_to_page = []
all_images = []


def parse_page(page_no, page):
    find_tables = page.find_tables(table_settings=table_settings)
    extract_tables = page.extract_tables(table_settings=table_settings)
    chars = page.chars
    images = page.images

    page_x0, page_y0, page_x1, page_y1 = (
        page.bbox[0],
        page.bbox[1],
        page.bbox[2],
        page.bbox[3],
    )

    if debug:
        print(f"---------- {page_no} page -----------------")
        print(f"find_tables: {len(find_tables)}")
        print(f"page size: {page_x1-page_x0} x {page_y1-page_y0}")
        print("---------------------------------")

    elements = []

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

    # 이미지 요소 추가
    for image in images:
        elements.append(
            {
                "type": "image",
                "x0": image["x0"],
                "y0": image["y0"],
                "x1": image["x1"],
                "y1": image["y1"],
                "content": image,
            }
        )

    # x0, y0로 정렬
    # elements.sort(key=lambda el: (-el["y0"], el["x0"]))

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

            table = is_within_tables(
                find_tables,
                extract_tables,
                text_x0,
                text_y0,
                text_x1,
                text_y1,
                # find_tables,
                # extract_tables,
                # x0,
                # y0,
                # x1,
                # y1,
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
                    # text += (
                    #     f" ({text_x0, text_y0, text_x1, text_y1}), ({x0, y0, x1, y1})\n"
                    # )
                    text += f" [text]({round(text_x0), round(text_y0), round(text_x1), round(text_y1)}), ({round(x0), round(y0), round(x1), round(y1)})\n"
                result.append(text)
        elif element["type"] == "image":
            image = element["content"]
            img_x0, img_y0, img_x1, img_y1 = (
                image["x0"],
                image["y0"],
                image["x1"],
                image["y1"],
            )

            if debug:
                # img_description = (
                #     f"[이미지 at ({img_x0}, {img_y0}, {img_x1}, {img_y1})]"
                # )
                img_description = f"[image] ({round(img_x0)}, {round(img_y0)}, {round(img_x1)}, {round(img_y1)})"
            else:
                img_description = "[이미지.png]"
            result.append("\n\n" + img_description + "\n\n")


for page_no, page in enumerate(pages):
    parse_page(page_no + 1, page)

print("".join(result))

# parse_page(1, page=pages[0])
# parse_page(2, page=pages[1])
# print("".join(result))
