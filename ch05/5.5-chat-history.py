from langchain_community.chat_message_histories import SQLChatMessageHistory


def respond(session_id: str, human_message: str) -> str:
    chat_message_history = SQLChatMessageHistory(
        session_id=session_id, connection="sqlite:///sqlite.db"
    )
    messages = chat_message_history.get_messages()

    ai_message = chain.invoke(
        {
            "chat_history": messages,
            "input": human_message,
        }
    )
    chat_message_history.add_user_message(human_message)
    chat_message_history.add_ai_message(ai_message)
    return ai_message


from uuid import uuid4

session_id = uuid4().hex
output1 = respond(
    session_id=session_id,
    human_message="こんにちは!私はジョンと言います!",
)
print(output1)
output2 = respond(
    session_id=session_id,
    human_message="私の名前が分かりますか?",
)
print(output2)
