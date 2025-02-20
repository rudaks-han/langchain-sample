import cv2
import pytesseract

# 이미지 파일 열기
image_path = "/tmp/img2.png"
img = cv2.imread(image_path)

# OCR로 텍스트 추출
text = pytesseract.image_to_string(img, lang="kor+eng")
print("텍스트 추출 결과:")
print(text)
