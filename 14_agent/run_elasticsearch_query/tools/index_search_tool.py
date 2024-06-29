import json

import tiktoken
from elasticsearch import Elasticsearch
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field  # <-- Uses v1 namespace

elasticsearch = Elasticsearch("http://172.16.120.203:9200")


class SearchToolInput(BaseModel):
    """Input for the index show data tool."""

    index_name: str = Field(
        ..., description="The name of the index for which the data is to be retrieved"
    )
    query: str = Field(
        ...,
        description="The ElasticSearch JSON query used to filter all hits. Should use the _source field if possible to specify required fields.",
    )
    from_: int = Field(
        ..., description="The record index from which the query will start"
    )
    size: int = Field(
        ...,
        description="How many records will be retrieved from the ElasticSearch query",
    )


def elastic_search(
    index_name: str,
    query: str,
    from_: int = cfg.elastic_index_data_from,
    size: int = cfg.elastic_index_data_size,
):
    """Executes a specific query on an ElasticSearch index and returns all hits or aggregation results"""
    size = min(cfg.elastic_index_data_max_size, size)
    encoding = tiktoken.encoding_for_model(cfg.model)
    try:
        full_dict: dict = json.loads(query)
        query_dict = None
        aggs_dict = None
        sort_dict = None
        if "query" in full_dict:
            query_dict = full_dict["query"]
        if "aggs" in full_dict:
            aggs_dict = full_dict["aggs"]
        if "sort" in full_dict:
            sort_dict = full_dict["sort"]
        if query_dict is None and aggs_dict is None and sort_dict is None:
            # Assume that there is a query but that the query part was ommitted.
            query_dict = full_dict
        if query_dict is None and aggs_dict is not None:
            # This is an aggregation query, therefore we suppress the hits here
            size = elasticsearch.aggs_limit
        print(query)

        final_res = ""
        retries = 0
        max_retries = 1
        while retries < max_retries:
            res = elasticsearch.search(
                index=index_name,
                from_=from_,
                size=size,
                query=query_dict,
                aggs=aggs_dict,
                sort=sort_dict,
            )
            if query_dict is None and aggs_dict is not None:
                # When a result has aggregations, just return that and ignore the rest
                final_res = str(res["aggregations"])
            else:
                final_res = str(res["hits"])
            tokens = encoding.encode(final_res)
            retries += 1
            if len(tokens) > cfg.token_limit:
                size -= 1
            else:
                return final_res

    except Exception as e:
        print(f"Could not execute query {query}")
        msg = str(e)
        return msg


def create_search_tool():
    return StructuredTool.from_function(
        elastic_search, name="elastic_index_search_tool", args_schema=SearchToolInput
    )


if __name__ == "__main__":
    index_name = "co2_production"
    tool = create_search_tool()
    res = tool.run(
        {
            "index_name": index_name,
            "query": """{"size": 0, "aggs" : { "regions" : { "terms" : { "field" : "region", "size": 100 } } } }""",
            "from_": 0,
            "size": 10,
        }
    )
