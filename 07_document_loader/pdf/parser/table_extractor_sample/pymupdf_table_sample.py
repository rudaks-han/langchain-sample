import pymupdf
import pymupdf4llm

file_path = "../../sample2/BM202404110000030261_0.pdf"
# file_path = "../../sample2/BM202405100000032919_0.pdf"
doc = pymupdf.open(file_path)

page = doc[0]

fonts = page.get_fonts()
tabs = page.find_tables(vertical_strategy="text", horizontal_strategy="text")
print(f"{len(tabs.tables)} table(s) on {page}")

# for tab in tabs:
#     print(tab.extract())


markdown = pymupdf4llm.to_markdown(file_path)
print(markdown)
