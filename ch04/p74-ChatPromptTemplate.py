from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザーが入力した料理のレシピを教えてください。"),
        ("human", "{dish}"),
    ]
)

prompt_value = prompt.invoke({"dish": "カレー"})
print(prompt_value)
