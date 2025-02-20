# ghostscript 설치 오류: https://yeonsikc.tistory.com/27
# pip install "camelot-py[base]"
import camelot

# file_path = "../../sample2/BM202404290000031873_0.pdf"
file_path = "../../sample2/BM202404110000030261_0.pdf"
tables = camelot.read_pdf(file_path, flavor="stream")

for table in tables:
    print(table)
