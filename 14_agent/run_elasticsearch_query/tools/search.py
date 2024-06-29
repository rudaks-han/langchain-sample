from elasticsearch import Elasticsearch
from langchain.tools import Tool

elasticsearch = Elasticsearch("http://172.16.120.203:9200")


def run_elasticsearch_query(query):
    resp = elasticsearch.get(index="logs", id=1)
    elasticsearch.search(index="logs_*", query={"match": {"foo": {"query": "foo"}}})

    print(resp["_source"])
    elasticsearch.search()
    # c = conn.cursor()
    # try:
    #     c.execute(query)
    #     return c.fetchall()
    # except sqlite3.OperationalError as err:
    #     return f"The following error occurred: {str(err)}"


run_query_tool = Tool.from_function(
    name="run_elasticsearch_query",
    description="Run a elasticsearch query.",
    func=run_elasticsearch_query,
)
