import pdfplumber
from pdfplumber.page import Page

file_path = "../../sample2/BM202404110000030261_0.pdf"
# file_path = "../../sample2/BM202404290000031873_0.pdf"
pdf = pdfplumber.open(file_path)
pages = pdf.pages


def debug_tablefinder(page: Page, table_settings):
    page_image = page.to_image()
    tablefinder = page_image.reset().debug_tablefinder(table_settings)
    tablefinder.save(f"debug_table-{page.page_number}.png")


for page in pages:
    # croped_page = pages[0].crop(
    #     (36.08979415893555, 960.8895874023438, 713.369873046875, 1373.876220703125)
    # )
    #
    # croped_page.to_image().save("table.png", format="PNG")

    # pdf = PDF("table.png", pages=[0], detect_rotation=False, pdf_text_extraction=True)
    table_settings = {
        # 텍스트 위치 기반으로 테이블 구조 파악
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        # 허용 오차값 증가로 텍스트 그룹화 개선
        # "snap_tolerance": 10,
        "snap_x_tolerance": 10,
        "snap_y_tolerance": 10,
        # "text_tolerance": 1,
        "text_x_tolerance": 10,
        "text_y_tolerance": 0,
        # "join_tolerance": 0,
        "join_x_tolerance": 0,
        "join_y_tolerance": 0,
        # 최소 단어 수 요구사항 조정
        "min_words_vertical": 0,  # 세로 방향 최소 단어 수 감소
        "min_words_horizontal": 10,  # 가로 방향은 최소 1개 유지
        "explicit_vertical_lines": [],
        "explicit_horizontal_lines": [],
        "edge_min_length": 0,
    }

    tables = page.extract_tables(table_settings=table_settings)

    find_tables = page.find_tables(table_settings=table_settings)

    for table in tables:
        # print(table)
        print("\n-----\n".join(["|".join(row) for row in table]))
        debug_tablefinder(page, table_settings)
    # print(find_tables)
    # words = page.extract_words()
    # table = page.extract_tables(table_settings)
    # print(table)

    table_finders = page.debug_tablefinder(table_settings=table_settings)
    pass


# "vertical_strategy": "text",
# "horizontal_strategy": "lines",
# "explicit_vertical_lines": [],
# "explicit_horizontal_lines": [],
# # "snap_tolerance": 3,
# # "snap_x_tolerance": 3,
# # "snap_y_tolerance": 3,
# # "join_tolerance": 3,
# # "join_x_tolerance": 3,
# "join_y_tolerance": 3,
# "edge_min_length": 3,
# # "min_words_vertical": 3,
# # "min_words_horizontal": 1,
# # "intersection_tolerance": 3,
# # "intersection_x_tolerance": 3,
# # "intersection_y_tolerance": 3,
# "text_tolerance": 3,
# "text_x_tolerance": 3,
# "text_y_tolerance": 3,
