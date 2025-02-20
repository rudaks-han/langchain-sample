import pdfplumber
from pdfplumber.page import Page
from pdfplumber.table import Table
from typing import Tuple, Dict, List


def get_table_column_count_not_empty(table: list):
    column_count = 0
    for row in table:
        row_column_count = 0
        for column in row:
            if column is not None and column != "":
                row_column_count += 1
        if row_column_count > column_count:
            column_count = row_column_count

    return column_count


def replace_newline_to_br(text: str):
    return text.replace("\n", "<br/>")


def table_to_markdown(table, table_index, table_merged_cells):
    markdown: str

    # table의 column이 1개 있는 경우 테이블로 만들지 않는다.
    column_count_not_empty = get_table_column_count_not_empty(table)
    # 헤더만 있는 경우 테이블로 만들지 않는다.
    table_row_count = len(table)

    if column_count_not_empty <= 1:
        markdown = ""
        for row_index, row in enumerate(table):
            markdown += "\n"
            for col_index, col in enumerate(row):
                if col is not None:
                    col = replace_newline_to_br(col)
                    markdown += col

        return "\n" + markdown + "\n"
    elif table_row_count == 1:
        markdown = ""
        for row_index, row in enumerate(table):
            markdown += "\n"
            for col_index, col in enumerate(row):
                if col is not None:
                    col = replace_newline_to_br(col)
                    markdown += " " + col

        return "\n" + markdown + "\n"
    else:
        headers = []
        header_first_value = None
        for value in table[0]:
            if header_first_value is None:
                header_first_value = value
            if value is None:
                value = header_first_value
                if value is None:
                    value = ""

            value = replace_newline_to_br(value)
            headers.append(value)
        header_count = len(headers)

        if header_count > 0:
            markdown = "| " + " | ".join(headers) + " |" + "\n"
            markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"

            prev_col_text: list = []  # 이전 행의 값 (row 기준으로 이전 row의 값)
            for row_index, row in enumerate(table[1:]):
                if all(col == "" or col is None for col in row):  # 모든 값이 공백이라면
                    continue

                curr_col_text: list = []  # 현재 행의 값
                for col_index, col in enumerate(row):
                    merged_cell = is_merged_cell(
                        table_merged_cells, col_index, row_index + 1
                    )
                    merged = merged_cell["merged"]

                    if col is None:
                        if merged:
                            merged_direction = merged_cell["direction"]
                            if merged_direction == "vertical":
                                if len(prev_col_text) > col_index:
                                    col = prev_col_text[col_index]
                                else:
                                    col = headers[col_index]
                            elif merged_direction == "horizontal":
                                col = curr_col_text[col_index - 1]
                            else:
                                if col_index > 0:
                                    col = curr_col_text[col_index - 1]
                                else:
                                    if len(prev_col_text) > col_index:
                                        col = prev_col_text[col_index]
                                    else:
                                        col = "[not defined]"
                        else:
                            col = ""

                    if col is None:
                        col = ""

                    col = col.replace("\n", "<br/>")
                    curr_col_text.append(col)

                markdown += "| " + " | ".join(curr_col_text) + " |\n"
                prev_col_text = curr_col_text

        else:
            markdown = ""
            for row in table:
                if row[0] is not None and row[0] != "":
                    markdown += row[0] + "\n"
        return "\n\n" + markdown + "\n\n"


def debug_tablefinder(page: Page, table_settings):
    page_image = page.to_image()
    tablefinder = page_image.reset().debug_tablefinder(table_settings)
    tablefinder.save(f"debug_table-{page.page_number}.png")


def check_merged_cells(
    cell_bbox: Tuple[float, float, float, float],
    row_col_mark: Dict[str, List[float]],
) -> Tuple[int, int, int, int]:
    cell = [round(x, 3) for x in cell_bbox]

    rows_y = row_col_mark["rows"]
    cols_x = row_col_mark["cols"]
    try:
        row_start = rows_y.index(cell[1])
        if cell[3] > rows_y[-1]:
            row_end = len(rows_y) - 1
        else:
            row_end = rows_y.index(cell[3]) - 1
        col_start = cols_x.index(cell[0])
        if cell[2] > cols_x[-1]:
            col_end = len(cols_x) - 1
        else:
            col_end = cols_x.index(cell[2]) - 1
        if not (row_end == row_start and col_end == col_start):
            return [row_start, row_end, col_start, col_end]

    except Exception as e:
        return None
    return None


def table_row_col_mark(table: Table) -> Dict[str, List[float]]:
    rows = table.rows
    rows_value = []
    cols_value = []
    for row in rows:
        for cell in row.cells:
            if cell is None:
                continue
            rows_value.append(cell[1])
            rows_value.append(cell[3])
            cols_value.append(cell[0])
            cols_value.append(cell[2])
    rows_value = sorted([round(x, 3) for x in set(rows_value)])
    cols_value = sorted([round(x, 3) for x in set(cols_value)])
    return {
        "rows": rows_value,
        "cols": cols_value,
    }


def extract_table_merged_cells(page: Page, table_settings={}):
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

    return table_merged_cells


def is_merged_cell(table_merged_cells, cell_index, row_index):
    for table_merged_cell in table_merged_cells:
        merged = table_merged_cell["merged"]

        y0, y1, x0, x1 = (
            merged[0],
            merged[1],
            merged[2],
            merged[3],
        )

        if y0 <= row_index <= y1 and x0 <= cell_index <= x1:
            # return True
            return {
                "merged": True,
                "direction": get_merged_cell_direction(x0, x1, y0, y1),
            }

    return {
        "merged": False,
    }


def get_merged_cell_direction(x0, x1, y0, y1):
    if x0 == x1:
        return "vertical"
    elif y0 == y1:
        return "horizontal"
    else:
        return "diagonal"


def is_within_bounds(x0, y0, x1, y1, bound_x0, bound_y0, bound_x1, bound_y1, char):
    # print(
    #     f"x0: {x0}, y0: {y0}, x1: {x1}, y1: {y1}, bound_x0: {bound_x0}, bound_y0: {bound_y0}, bound_x1: {bound_x1}, bound_y1: {bound_y1}"
    # )

    text = char["text"]
    if text == "시간":
        pass

    return (bound_x0 <= x0 <= bound_x1 and bound_y0 <= y0 <= bound_y1) and (
        bound_x0 <= x1 <= bound_x1 and bound_y0 <= y1 <= bound_y1
    )


def is_within_tables(find_tables, extract_tables, x0, y0, x1, y1, char) -> Table:

    for table_index, table in enumerate(find_tables):
        table_x0, table_y0, table_x1, table_y1 = table.bbox

        if is_within_bounds(
            x0, y0, x1, y1, table_x0, table_y0, table_x1, table_y1, char
        ):
            return table_index, extract_tables[table_index]

    return -1, None


def is_within_tables_ocr(ocr_extract_tables, x0, y0, x1, y1, char) -> Table:

    for table_index, table in enumerate(ocr_extract_tables):
        text = char["text"]
        if text == "상담회":
            pass
        table_x0, table_x1, table_y0, table_y1 = sorted(table["bbox"])

        if is_within_bounds(
            x0, y0, x1, y1, table_x0, table_y0, table_x1, table_y1, char
        ):
            return table_index, ocr_extract_tables[table_index]

    return -1, None


def is_text_within_table(char: dict, find_tables, extract_tables):
    x0, top, x1, bottom = (
        char["x0"],
        char["top"],
        char["x1"],
        char["bottom"],
    )

    return is_within_tables(find_tables, extract_tables, x0, top, x1, bottom, char)


def is_text_within_table_ocr(char: dict, ocr_extracted_tables):
    x0, top, x1, bottom = (
        char["x0"],
        char["top"],
        char["x1"],
        char["bottom"],
    )

    return is_within_tables_ocr(
        ocr_extracted_tables,
        x0,
        top,
        x1,
        bottom,
        char,
    )


def curves_to_edges(cs):
    edges = []
    for c in cs:
        edges += pdfplumber.utils.rect_to_edges(c)
    return edges
