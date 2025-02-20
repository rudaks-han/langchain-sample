from pdfminer.high_level import extract_text

# file_path = "../../sample2/BM202403270000028996_0.pdf"
file_path = "../../sample2/BM202404110000030261_0.pdf"

text = extract_text(file_path)
print(text)
