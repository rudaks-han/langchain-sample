# pip install jpype1
import os

os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/zulu-21.jdk/Contents/Home"


from tabula import read_pdf

file_path = "../../sample2/BM202404110000030261_0.pdf"
# file_path = "../../sample2/BM202404290000031873_0.pdf"

tables = read_pdf(file_path, pages="1", stream=True, lattice=True)
for table in tables:
    print(table)


# dfs = tabula.read_pdf(file_path, lattice=False)
# print(f"Data Type :{type(dfs)}")
# print(f"Data Length: {len(dfs)}")
# for index, table in enumerate(dfs):
#     print(f"\nData Index: {index}")
#     print(type(table))
#     print(table.head())
