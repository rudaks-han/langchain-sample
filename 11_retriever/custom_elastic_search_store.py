from typing import List, Optional, Dict, Any, Callable

from langchain_core.documents import Document
from langchain_elasticsearch.vectorstores import (
    ElasticsearchStore,
    _hits_to_docs_scores,
)


class CustomElasticSearchStore(ElasticsearchStore):
    def custom_script_query(self, query_body: dict, query: str):
        query_vector = query_body["knn"]["query_vector"]
        _filter = query_body["knn"]["filter"]
        must_clauses = []

        if _filter:
            for key, value in _filter.items():
                must_clauses.append({"match": {f"metadata.{key}.keyword": f"{value}"}})
        else:
            must_clauses.append({"match_all": {}})

        return {
            "query": {
                "script_score": {
                    "query": {
                        "bool": {"must": must_clauses},
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                        "params": {"query_vector": query_vector},
                    },
                }
            }
        }

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 50,
        filter: Optional[List[dict]] = None,
        *,
        custom_query: Optional[
            Callable[[Dict[str, Any], Optional[str]], Dict[str, Any]]
        ] = None,
        doc_builder: Optional[Callable[[Dict], Document]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        hits = self._store.search(
            query=query,
            k=k,
            num_candidates=fetch_k,
            filter=filter,
            custom_query=self.custom_script_query,
        )
        docs = _hits_to_docs_scores(
            hits=hits,
            content_field=self.query_field,
        )
        return [doc for doc, _score in docs]
