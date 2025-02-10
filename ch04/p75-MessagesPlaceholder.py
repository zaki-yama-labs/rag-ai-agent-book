from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Your are a helpful assistant."),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
    ]
)

prompt_value = (
    prompt.invoke(
        {
            "chat_history": [
                HumanMessage(content="こんにちは！私はジョンと言います！"),
                AIMessage("こんにちは、ジョンさん！どのようにお手伝いできますか？"),
            ],
            "input": "私の名前がわかりますか？",
        }
    ),
)

print(prompt_value)
