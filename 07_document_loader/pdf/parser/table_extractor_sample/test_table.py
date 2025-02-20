import pdfplumber

file_path = "../../sample2/BM202404110000030261_0.pdf"

pdf = pdfplumber.open(file_path)
page = pdf.pages[0]

table_settings = {
    "vertical_strategy": "text",
    "horizontal_strategy": "lines",
    "explicit_vertical_lines": [],
    "explicit_horizontal_lines": [],
    # "snap_tolerance": 3,
    # "snap_x_tolerance": 3,
    # "snap_y_tolerance": 3,
    # "join_tolerance": 3,
    # "join_x_tolerance": 3,
    "join_y_tolerance": 3,
    "edge_min_length": 3,
    # "min_words_vertical": 3,
    # "min_words_horizontal": 1,
    # "intersection_tolerance": 3,
    # "intersection_x_tolerance": 3,
    # "intersection_y_tolerance": 3,
    "text_tolerance": 3,
    "text_x_tolerance": 3,
    "text_y_tolerance": 3,
}
tables = page.extract_tables(table_settings)
print(tables)
