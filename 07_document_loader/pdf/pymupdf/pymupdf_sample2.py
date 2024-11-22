import pymupdf

# file_path = "./sample/SPRI_AI_Brief_2023년12월호_F.pdf"
# file_path = "../sample/invoice.pdf"
# file_path = "./sample/invoice_sample.pdf"
file_path = "../sample/table_sample.pdf"
# file_path = "../sample/샘플.pdf"

doc = pymupdf.open(file_path)


def extract_pdf_content(pdf_path):
    # PDF 문서 열기
    doc = pymupdf.open(pdf_path)

    # 결과를 저장할 변수들
    full_text = ""
    page_texts = {}
    tables = []

    # 각 페이지 처리
    for page_num in range(len(doc)):
        page = doc[page_num]

        # 텍스트 추출
        text = page.get_text()
        page_texts[page_num + 1] = text
        full_text += text + "\n\n"

        # 표 추출
        # 표를 찾기 위해 테이블 구조를 가진 블록 검색
        blocks = page.get_text("blocks")
        for block in blocks:
            # 블록이 표 형태인지 확인 (4개 이상의 직사각형이 정렬된 형태)
            if is_table_block(block):
                table_data = extract_table_from_block(block)
                if table_data:
                    tables.append({"page": page_num + 1, "data": table_data})

    doc.close()
    return full_text, page_texts, tables


def is_table_block(block):
    """
    블록이 표 형태인지 확인하는 함수
    """
    # 블록의 구조를 분석하여 표 형태인지 확인
    x0, y0, x1, y1, text, block_type, *_ = block

    # 간단한 휴리스틱:
    # 1. 여러 줄의 텍스트가 있고
    # 2. 텍스트에 구분자(|, \t 등)가 있거나
    # 3. 일정한 간격으로 정렬된 텍스트가 있는 경우
    lines = text.split("\n")
    if len(lines) < 2:
        return False

    # 구분자 확인
    has_separators = any("|" in line or "\t" in line for line in lines)

    # 텍스트 정렬 패턴 확인
    has_alignment = check_text_alignment(lines)

    return has_separators or has_alignment


def check_text_alignment(lines):
    """
    텍스트 라인들이 정렬된 패턴을 가지는지 확인
    """
    if not lines:
        return False

    # 각 라인의 단어 수가 일정한지 확인
    words_per_line = [len(line.split()) for line in lines]
    return len(set(words_per_line)) == 1 and words_per_line[0] > 1


def extract_table_from_block(block):
    """
    블록에서 표 데이터를 추출하는 함수
    """
    x0, y0, x1, y1, text, block_type, *_ = block

    # 텍스트를 라인으로 분리
    lines = text.split("\n")
    if not lines:
        return None

    # 표 데이터 구성
    table_data = []
    for line in lines:
        # 구분자가 있는 경우
        if "|" in line:
            row = [cell.strip() for cell in line.split("|")]
        # 탭으로 구분된 경우
        elif "\t" in line:
            row = [cell.strip() for cell in line.split("\t")]
        # 공백으로 구분된 경우
        else:
            row = line.split()

        if row:
            table_data.append(row)

    return table_data if table_data else None


# 사용 예시
def main():
    full_text, page_texts, tables = extract_pdf_content(file_path)

    print(full_text)
    # # 전체 텍스트 출력
    # print("전체 텍스트:")
    # print(full_text[:500] + "...\n")
    #
    # # 페이지별 텍스트 출력
    # print("페이지별 텍스트:")
    # for page_num, text in page_texts.items():
    #     print(f"페이지 {page_num}:")
    #     print(text[:200] + "...\n")
    #
    # # 추출된 표 출력
    # print("추출된 표:")
    # for table in tables:
    #     print(f"페이지 {table['page']}의 표:")
    #     df = pd.DataFrame(table["data"])
    #     print(df)
    #     print("\n")


if __name__ == "__main__":
    main()
