# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

# 웹 기반 문서 로더를 초기화합니다.
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")

# 문서를 로드합니다.
docs = loader.load()


class StreamCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        print(f"{token}", end="", flush=True)


# OpenAI의 Chat 모델을 초기화합니다. 여기서는 온도를 0으로 설정하고 모델 이름을 지정합니다.
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo-16k",
    streaming=True,
    callbacks=[StreamCallback()],
)
# 요약 체인을 로드합니다. 체인 타입을 'stuff'로 지정합니다.
chain = load_summarize_chain(llm, chain_type="stuff")

# 문서에 대해 요약 체인을 실행합니다.
answer = chain.invoke({"input_documents": docs})
print(answer["output_text"])


from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain import hub


def stuff():
    # 원격 저장소에서 프롬프트를 가져오는 경우
    prompt = hub.pull("teddynote/summary-stuff-documents-korean")

    # LLM 체인 정의
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo-16k",
        streaming=True,
        callbacks=[StreamCallback()],
    )

    # LLMChain 정의
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # StuffDocumentsChain 정의
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="context"
    )

    docs = loader.load()
    response = stuff_chain.invoke({"input_documents": docs})
    print(response)


def map_reduce():
    from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains.llm import LLMChain
    from langchain import hub

    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        streaming=True,
        callbacks=[StreamCallback()],
    )

    # # map-prompt 를 직접 정의하는 경우 다음의 예시를 참고하세요.
    # map_template = """The following is a set of documents
    # {docs}
    # Based on this list of docs, please identify the main themes
    # Helpful Answer:"""
    # map_prompt = PromptTemplate.from_template(map_template)

    # langchain 허브에서 'rlm/map-prompt'를 가져옵니다.
    map_prompt = hub.pull("teddynote/map-prompt")
    print(map_prompt)

    # LLMChain 인스턴스를 생성하며, 이때 LLM과 프롬프트로 'map_prompt'를 사용합니다.
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    reduce_prompt = hub.pull("teddynote/reduce-prompt-korean")
    print(reduce_prompt)

    from langchain.chains.combine_documents.stuff import StuffDocumentsChain

    # 연쇄 실행
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # 문서 리스트를 받아 하나의 문자열로 결합한 후 LLMChain에 전달
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries"
    )

    # 매핑된 문서들을 결합하고 반복적으로 축소
    reduce_documents_chain = ReduceDocumentsChain(
        # 최종적으로 호출되는 체인입니다.
        combine_documents_chain=combine_documents_chain,
        # `StuffDocumentsChain`의 컨텍스트를 초과하는 문서들을 처리
        collapse_documents_chain=combine_documents_chain,
        # 문서들을 그룹화할 최대 토큰 수.
        token_max=4096,
    )

    # 문서들을 매핑하여 체인을 거친 후 결과를 결합하는 과정
    map_reduce_chain = MapReduceDocumentsChain(
        # 매핑 체인
        llm_chain=map_chain,
        # 리듀스 체인
        reduce_documents_chain=reduce_documents_chain,
        # llm_chain에서 문서들을 넣을 변수 이름
        document_variable_name="docs",
        # 매핑 단계의 결과를 출력에 포함시킴
        return_intermediate_steps=False,
    )

    # 문자를 기준으로 텍스트를 분할하는 객체 생성
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        length_function=len,
    )

    # 문서들을 분할
    split_docs = text_splitter.split_documents(docs)

    # split_docs를 map_reduce_chain의 run 메서드에 전달하여 실행한 결과를 출력합니다.
    summary_result = map_reduce_chain.invoke({"input_documents": split_docs})

    print(summary_result["output_text"])


# def refine():
#     # llm을 사용하여 'refine' 유형의 요약 체인을 로드합니다.
#     chain = load_summarize_chain(llm, chain_type="refine")
#     # split_docs를 처리하기 위해 체인을 실행합니다.
#     chain.run(split_docs)
#
#     prompt_template = """Write a concise summary of the following:
# {text}
# CONCISE SUMMARY:"""
#     prompt = PromptTemplate.from_template(prompt_template)
#
#     refine_template = (
#         "Your job is to produce a final summary\n"
#         "We have provided an existing summary up to a certain point: {existing_answer}\n"
#         "We have the opportunity to refine the existing summary"
#         "(only if needed) with some more context below.\n"
#         "------------\n"
#         "{text}\n"
#         "------------\n"
#         "Given the new context, refine the original summary in Korean"
#         "If the context isn't useful, return the original summary."
#     )
#     refine_prompt = PromptTemplate.from_template(refine_template)
#     chain = load_summarize_chain(
#         llm=llm,
#         chain_type="refine",
#         question_prompt=prompt,
#         refine_prompt=refine_prompt,
#         return_intermediate_steps=True,
#         input_key="input_documents",
#         output_key="output_text",
#     )
#     result = chain.invoke({"input_documents": split_docs}, return_only_outputs=True)
#
#     print(
#         result["output_text"]
#     )  # 결과 딕셔너리에서 'output_text' 키에 해당하는 값을 출력합니다.
#
#     print("\n\n".join(result["intermediate_steps"][:3]))
