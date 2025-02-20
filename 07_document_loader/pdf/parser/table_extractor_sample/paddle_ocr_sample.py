from paddleocr import PaddleOCR

# PaddleOCR 초기화
ocr = PaddleOCR(use_angle_cls=False, lang="korean")  # 'ko'로 언어 설정

# 이미지 경로
image_path = "./sample/img.png"

# OCR 수행
result = ocr.ocr(image_path)
print(result)

# img = Image.open(image_path)
# text = pytesseract.image_to_string(img, lang="kor")
# print(text)

# 결과 출력 및 테이블 생성
# table_data = []
# for line in result[0]:
#     text = line[1][0]  # 인식된 텍스트
#     table_data.append([text])

# Pandas DataFrame으로 변환
# df = pd.DataFrame(table_data, columns=["텍스트"])
# print(df)
