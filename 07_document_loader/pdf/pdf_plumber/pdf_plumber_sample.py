import pdfplumber
from pdfplumber.table import Table

# file_path = "./sample/SPRI_AI_Brief_2023년12월호_F.pdf"
# file_path = "./sample/invoice_sample.pdf"
# file_path = "./sample/invoice.pdf"
# file_path = "./sample/table_sample.pdf"
file_path = "./sample/샘플.pdf"


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
    markdown_table = "| " + " | ".join(filtered_row) + " |" + "\n"
    markdown_table += "| " + " | ".join(["---"] * len(filtered_row)) + " |\n"

    for row in table[1:]:
        column = []
        for col_index, col in enumerate(row):
            if col != "":
                column.append(col)
        markdown_table += "| " + " | ".join(column) + " |\n"

    return markdown_table


table_settings = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
}

pdf = pdfplumber.open(file_path)
pages = pdf.pages

result = []

table_append_to_page = []
all_images = []

for page in pages:
    find_tables = page.find_tables(table_settings=table_settings)
    extract_tables = page.extract_tables(table_settings=table_settings)
    chars = page.chars
    images = page.images
    print("=== text ===")
    for char in chars:
        text_x0, text_y0, text_x1, text_y1 = (
            char["doctop"],
            char["top"],
            char["doctop"] + char["height"],
            char["bottom"],
        )
        x0, y0, x1, y1 = (
            char["x0"],
            char["y0"],
            char["x1"],
            char["y1"],
        )

        table = is_within_tables(
            find_tables, extract_tables, text_x0, text_y0, text_x1, text_y1
        )
        if table is not None:
            markdown_table = table_to_markdown(table)
            if markdown_table not in table_append_to_page:
                result.append("\n\n")
                result.append(markdown_table)
                table_append_to_page.append(markdown_table)
        else:
            text = char["text"]
            text += f" ({text_x0, text_y0, text_x1, text_y1}), ({x0, y0, x1, y1})"
            result.append(text)

    # for image in page.images:
    #     all_images.append(image)
    # print("=== tables ===")
    # tables = page.find_tables()
    # tables = page.extract_tables()
    # print(tables)

# print(result)
print("\n".join(result))


for page in pdf.pages:
    for image in page.images:
        # x0, y0, x1, y1 = image
        text_x0, text_y0, text_x1, text_y1 = (
            char["x0"],
            char["y0"],
            char["x1"],
            char["y1"],
        )
        print(f"image: {text_x0, text_y0, text_x1, text_y1}")

# (72.24000000000001, 119.75999999999999, 522.96, 173.27999999999997)

# page = pages[0].extract_tables()
# print(page)
