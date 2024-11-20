from langchain_community.document_loaders import PDFPlumberLoader

file_path = "./sample/SPRI_AI_Brief_2023년12월호_F.pdf"
loader = PDFPlumberLoader(file_path)


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"[doc {i+1}]" + d.page_content for i, d in enumerate(docs)]
        )
    )


if __name__ == "__main__":
    docs = loader.load()
    pretty_print_docs(docs)
