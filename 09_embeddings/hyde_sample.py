import langchain
from dotenv import load_dotenv
from langchain.chains import HypotheticalDocumentEmbedder, LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI, OpenAIEmbeddings

langchain.debug = True

load_dotenv()

base_embeddings = OpenAIEmbeddings()
llm = OpenAI()

embeddings = HypotheticalDocumentEmbedder.from_llm(llm, base_embeddings, "web_search")

result = embeddings.embed_query("역대 프로야구에서 최고의 선수는?")

print(result)

# multi_llm = OpenAI(n=4, best_of=4)
# embeddings = HypotheticalDocumentEmbedder.from_llm(
#     multi_llm, base_embeddings, "web_search"
# )
# result = embeddings.embed_query("역대 프로야구에서 최고의 선수는?")
# print(result)
