from pdf_layout_parser import PdfLayoutParser

# file_path = "/Users/rudaks/temp/pdf/샘플.pdf"
# file_path = "../sample/table_sample.pdf"
# file_path = "../sample2/BM202403270000028996_0.pdf"
# file_path = "../sample2/BM202404010000029365_0.pdf"
# file_path = "../sample2/BM202404010000029381_0.pdf"
# file_path = "../sample2/BM202404040000029709_0.pdf"
# file_path = "../sample2/BM202404100000030137_0.pdf"
file_path = "../sample2/BM202404110000030261_0.pdf"
# 첫번째 테이블 추출 안됨. (표의 가로선이 점선, 세로는 없음)
# 나머지는 테이블 추출되지만 데이터가 깨짐

# file_path = "../sample2/BM202404180000030755_0.pdf"
# file_path = "../sample2/BM202404240000031264_0.pdf"
# file_path = "../sample2/BM202404290000031873_0.pdf"
# 표 문제 # 잘안됨

# file_path = "../sample2/BM202404290000031875_0.pdf"
# 잘안됨, font-size에 따른 띄어쓰기 필요


# file_path = "../sample2/BM202405030000032265_0.pdf"
# file_path = "../sample2/BM202405070000032410_0.pdf"
# file_path = "../sample2/BM202405070000032481_0.pdf"
# file_path = "../sample2/BM202405080000032560_0.pdf"  # 전체 이미지라서 불가
# file_path = "../sample2/BM202405080000032563_0.pdf"  # 전체 이미지라서 불가
# file_path = "../sample2/BM202405080000032619_0.pdf"  # 일부 확인 필요 (테이블 위치)
# file_path = "../sample2/BM202405080000032628_0.pdf"  # 일부 확인 필요 (header font-size)
# file_path = "../sample2/BM202405090000032671_0.pdf"
# 잘안됨

# file_path = "../sample2/BM202405090000032702_0.pdf"
# file_path = "../sample2/BM202405090000032712_0.pdf"  # 전체 이미지라서 불가
# file_path = "../sample2/BM202405090000032714_0.pdf"
# file_path = "../sample2/BM202405090000032728_0.pdf"
# 일부 확인 필요 (테이블 파싱 확인)

# file_path = "../sample2/BM202405100000032835_0.pdf"
# file_path = "../sample2/BM202405100000032919_0.pdf"
# 가로로 되어 있는 문서

# file_path = "../sample2/BM202405130000033013_0.pdf"
# file_path = "../sample2/BM202405140000033098_0.pdf"
# file_path = "../sample2/BM202405140000033101_0.pdf"
# file_path = "../sample2/BM202405140000033102_0.pdf"
# 확인 필요 (테이블이 복잡함)

# file_path = "../sample2/BM202405140000033115_0.pdf"
# 잘안됨

# file_path = "../sample2/BM202405140000033117_0.pdf"
# 확인 필요 (복잡한 테이블)

# file_path = "../sample2/background-checks.pdf"
# 엄청 복잡한 표 (가로로 되어 있음)

# file_path = "../sample2/ag-energy-round-up-2017-02-24.pdf"
# 불가

# file_path = "../sample2/ca-warn-report.pdf"
# file_path = "../sample2/san-jose-pd-firearm-sample.pdf"
# 불가

# file_path = "../sample2/tdostats.pdf"
# file_path = "../sample2/SPRI_AI_Brief_2023년12월호_F.pdf"
# file_path = "../sample2/2201.00069.pdf"
# file_path = "../sample2/battery-file-22.pdf"
# file_path = "../sample2/bug1945.pdf"
# file_path = "../sample2/chinese-tables.pdf"
# file_path = "../sample2/dotted-gridlines.pdf"
# file_path = "../sample2/test-2333.pdf"


if __name__ == "__main__":
    pdf_parser = PdfLayoutParser(file_path, debug=True)
    docs = pdf_parser.execute()

    result = ""
    for doc in docs:
        result += doc.page_content
    pdf_parser.save_to_markdown(result, "output.md")
    print(result)
