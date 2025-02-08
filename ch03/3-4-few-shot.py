from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "入力がAIに関係するか回答してください。"},
        {"role": "user", "content": "AIの進化はすごい"},
        {"role": "assistant", "content": "true"},
        {"role": "user", "content": "今日は良い天気だ"},
        {"role": "assistant", "content": "false"},
        {"role": "user", "content": "ChatGPTはとても便利だ"},
    ],
)
print(response.choices[0].message.content)
