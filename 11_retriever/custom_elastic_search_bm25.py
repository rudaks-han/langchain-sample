from typing import List

from langchain_community.retrievers import ElasticSearchBM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document


class CustomElasticSearchBM25Retriever(ElasticSearchBM25Retriever):
    search_args = {}

    def __init__(self, **search_args):
        super().__init__(**search_args)
        self.search_args = search_args

    def _get_relevant_documents(
        self, query: str, run_manager: CallbackManagerForRetrieverRun, **kwargs
    ) -> List[Document]:
        query_dict = {"query": {"match": {"content": query}}}

        size = -1
        if "search_args" in self.search_args and "k" in self.search_args["search_args"]:
            size = self.search_args["search_args"]["k"]

        res = self.client.search(index=self.index_name, body=query_dict)

        docs = []
        for i, r in enumerate(res["hits"]["hits"]):
            docs.append(Document(page_content=r["_source"]["content"]))
            if -1 < size <= i + 1:
                break
        return docs
