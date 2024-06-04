# pipenv install langchain-text-splitters tiktoken sentence-transformers


from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

# [print(e.value) for e in Language]

PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
docs = python_splitter.create_documents([PYTHON_CODE])
for i, doc in enumerate(docs):
    print(f"__{i}__")
    print(doc.page_content)
