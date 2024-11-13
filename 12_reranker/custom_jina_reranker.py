from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict
from transformers import AutoModelForSequenceClassification


class CustomJinaRerank(BaseDocumentCompressor):
    _model_name: str = "jina-reranker-v2-base-multilingual"
    top_n: Optional[int] = 3
    model: AutoModelForSequenceClassification = (
        AutoModelForSequenceClassification.from_pretrained(
            f"jinaai/{_model_name}",
            torch_dtype="auto",
            trust_remote_code=True,
        )
    )
    model.to("cpu")  # cuda/cpu
    model.eval()

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    def rerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        model_name: Optional[str] = None,
        top_n: Optional[int] = -1,
        max_chunks_per_doc: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if len(documents) == 0:  # to avoid empty api call
            return []
        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]
        top_n = top_n if (top_n is None or top_n > 0) else self.top_n

        results = self.model.rerank(
            query, docs, max_query_length=512, max_length=1024, top_n=top_n
        )

        result_dicts = []
        for res in results:
            result_dicts.append(
                {"index": res["index"], "relevance_score": res["relevance_score"]}
            )
        return result_dicts

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        compressed = []
        for res in self.rerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)
        return compressed
