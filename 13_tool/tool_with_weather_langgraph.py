import langchain
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

langchain.debug = True

load_dotenv()


@tool
def get_weather(location: str) -> str:
    """Get weather condition."""
    # 날씨 api 호출하는 로직이 구현한다.
    # 여기서는 api 호출 대신, 간단한 문자열을 리턴한다.
    result = f"{location}의 날씨는 맑음입니다."

    return result


llm = ChatOpenAI(model="gpt-4o-mini")
tools = [get_weather]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

query = "서울의 날씨는?"
result = tool_node.invoke({"messages": [llm_with_tools.invoke(query)]})
# "content='서울의 날씨는 맑음입니다.' name='get_weather' tool_call_id='call_IiUh7hfriCkGIU8U55fpyWe1'"
# {'messages': [ToolMessage(content='서울의 날씨는 맑음입니다.', name='get_weather', tool_call_id='call_IiUh7hfriCkGIU8U55fpyWe1')]}
print(result)
