import pytesseract
from PIL import Image

# 이미지 파일로부터 텍스트 추출
image = Image.open("./image002.png")
# image = Image.open("./목차.png")
text = pytesseract.image_to_string(image, lang="kor+eng")

print("____________")
print(text)
