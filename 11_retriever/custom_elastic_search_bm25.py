from typing import List

from langchain_community.retrievers import ElasticSearchBM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document


class CustomElasticSearchBM25Retriever(ElasticSearchBM25Retriever):
    def __init__(self, **search_args):
        super.__init__()
        self.search_args = search_args

    def _get_relevant_documents(
        self, query: str, run_manager: CallbackManagerForRetrieverRun, **kwargs
    ) -> List[Document]:
        query_dict = {"query": {"match": {"content": query}}}

        size = -1
        if "kwargs" in kwargs and "k" in kwargs["kwargs"]:
            size = kwargs["kwargs"]["k"]

        res = self.client.search(index=self.index_name, body=query_dict)

        docs = []
        for i, r in enumerate(res["hits"]["hits"]):
            docs.append(Document(page_content=r["_source"]["content"]))
            if size > -1 and i >= kwargs["k"]:
                break
        return docs
