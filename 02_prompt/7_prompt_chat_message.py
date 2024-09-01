from langchain_core.prompts import ChatMessagePromptTemplate

prompt = "나는 {country}로 여행가고 싶어"

chat_message_prompt = ChatMessagePromptTemplate.from_template(
    role="Steve", template=prompt
)
prompt_format = chat_message_prompt.format(country="한국")
print(prompt_format)
