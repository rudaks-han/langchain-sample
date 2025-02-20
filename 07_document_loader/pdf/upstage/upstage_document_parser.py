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
# file_path = "../sample/SPRI_AI_Brief_2023년12월호_F.pdf"
# file_path = "../sample/invoice.pdf"
# file_path = "../sample/invoice_sample.pdf"
# file_path = "../sample/table_sample.pdf"
# file_path = "../sample/ag-energy-round-up-2017-02-24.pdf"
# file_path = "../sample2/BM202403270000028996_0.pdf"
# file_path = "../sample2/BM202404100000030137_0.pdf"
# file_path = "../sample2/BM202404240000031264_0.pdf"
# file_path = "../sample2/BM202405080000032560_0.pdf"
# file_path = "../sample2/background-checks.pdf"
# file_path = "../sample2/BM202405030000032265_0.pdf"  # 일부 확인 필요
# file_path = "../sample2/BM202405080000032619_0.pdf"  # 일부 확인 필요
# file_path = "../sample2/BM202404110000030261_0.pdf"
# file_path = "../sample2/BM202404290000031873_0.pdf"
# file_path = "../sample2/BM202404290000031875_0.pdf"
# file_path = (
#     "../sample2/[스페션 1부] 개발자의 눈으로 바라본 SaaS (p32115).pdf"
# )
file_path = "../sample2/세로_split.pdf"

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
