from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("こんにちは!私はジョンと言います!"),
    AIMessage(content="こんにちは、ジョンさん!どのようにお手伝いできますか?"),
    HumanMessage(content="私の名前がわかりますか?"),
]

# ai_message = model.invoke(messages)
# print(ai_message.content)

for chunk in model.stream(messages):
    print(chunk.content, end="", flush=True)
