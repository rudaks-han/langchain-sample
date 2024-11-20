import os

from dotenv import load_dotenv
from langchain_upstage import UpstageDocumentParseLoader

load_dotenv()


def pretty_print_docs(docs):
    # print(
    #     f"\n{'-' * 100}\n".join(
    #         [f"[doc {i+1}]" + d.page_content for i, d in enumerate(docs)]
    #     )
    # )
    print(f"\n".join([d.page_content for i, d in enumerate(docs)]))


api_key = os.environ["UPSTAGE_API_KEY"]
file_path = "./sample/SPRI_AI_Brief_2023년12월호_F.pdf"

# url = "https://api.upstage.ai/v1/document-ai/document-parse"
# headers = {"Authorization": f"Bearer {api_key}"}
#
# files = {"document": open(filename, "rb")}
# response = requests.post(url, headers=headers, files=files, output_format="json")
#
# print(response.json())

layzer = UpstageDocumentParseLoader(file_path, split="page", output_format="markdown")

if __name__ == "__main__":
    docs = layzer.load()
    pretty_print_docs(docs)
    # for doc in docs[:3]:
    #     print(doc)
