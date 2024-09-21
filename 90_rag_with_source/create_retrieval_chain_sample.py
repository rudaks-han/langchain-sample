from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

embeddings = OpenAIEmbeddings()
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

query = "한모 씨의 여름 휴가는 언제야?"

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = vector_store.as_retriever()
#
template = """Answer the question based only on the following context:
{context}

Question: {input}
"""
prompt = ChatPromptTemplate.from_template(template)

question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

result = rag_chain.invoke(
    {
        "input": query,
    }
)
print(result)
# {'input': '한모 씨의 여름 휴가는 언제야?', 'context': [Document(metadata={'source': './data/news.txt'}, page_content='40대 직장인 한모 씨는 올해 여름휴가를 예년보다 앞당겨 6월에 쓰기로 했다. 평소 7~8월에 휴가를 다녀온 그는 "어딜 가도 사람 많고 더운 7~8월을 피해 6월에 다녀오려 한다. 마침 항공권도 구했고 비용도 성수기보다는 싸서 큰 마음 먹고 결정했다"고 말했다.'), Document(metadata={'source': './data/news.txt'}, page_content="6월이 '이른' 여행 성수기로 떠오르는 셈이다. 올해도 6월 여행을 떠나는 관광객이 늘어날 것으로 예상된다. 현충일(6월6일) 이튿날에 하루만 휴가를 사용하면 연이어 나흘을 쉴 수 있는 황금연휴가 있는 데다 여행업계도 각종 할인 프로그램을 내놓으면서다."), Document(metadata={'source': './data/news.txt'}, page_content="3일 문화체육관광부가 발표한 '2023년 국민여행조사'에 따르면 관광·휴양 목적으로 여행을 떠나는 국내 관광여행 횟수는 2023년 6월 2122만회로 전년 동월(2022년 6월 2044만회) 대비 3.8% 증가했다. 반면 여름휴가 성수기인 7~8월은 각각 2203만회(0.7% 증가), 2316만회(0.9% 감소)로 1년 전에 비해 소폭 증가하거나 감소한 것으로 나타났다."), Document(metadata={'source': './data/news.txt'}, page_content="특히 6월 한 달간 진행되는 '대한민국 숙박 세일페스타'는 국내 여행 수요를 이끌어 낼 것으로 보인다. 앞서 지난 2월과 3월 배포한 숙박 할인권은 여행 지출액 약 862억원, 지역 관광객 약 48만명 유발 효과를 낸 것으로 집계됐다. G마켓의 경우 전년 동월 대비 국내 여행 판매량이 숙박 할인권이 배포된 2월(97% 증가)과 3월(90% 증가)에 거의 2배 뛰었다.")], 'answer': '한모 씨의 여름 휴가는 6월에 계획되어 있습니다.'}
