from langchain_community.document_loaders import PyPDFLoader


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"[doc {i+1}]" + d.page_content for i, d in enumerate(docs)]
        )
    )


file_path = "./sample/SPRI_AI_Brief_2023년12월호_F.pdf"

# pip install rapidocr-onnxruntime
if __name__ == "__main__":
    loader = PyPDFLoader(file_path, extract_images=False, extraction_mode="layout")
    docs = loader.load()
    pretty_print_docs(docs)
