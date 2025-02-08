import json


def get_current_weather(location, unit="fahrenheit"):
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": unit}
        )
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

from openai import OpenAI

client = OpenAI()

messages = [
    {"role": "user", "content": "東京の天気はどうですか?"},
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools
)
print(response.to_json(indent=2))

# {
#   "id": "chatcmpl-9jj2vw14w0fYHr6CUAhfeoMW2ZlnI",
#   "choices": [
#     {
#       "finish_reason": "tool_calls", "index": 0,
#       "logprobs": null,
#       "message": {
#         "content": null,
#         "role": "assistant",
#         "tool_calls": [
#           {
#             "id": "call_if3ni88ahchW2egXXtZXnXA7",
#             "function": {
#               "arguments": "{\"location\":\"Tokyo\"}",
#               "name": "get_current_weather"
#             },
#             "type": "function"
#           }
#         ]
#       }
#     }
#   ],
#   ...
# }

response_message = response.choices[0].message
messages.append(response_message.to_dict())

available_functions = {
    "get_current_weather": get_current_weather,
}

# 使いたい関数は複数あるかもしれないのでループ
for tool_call in response_message.tool_calls:
    # 関数を実行
    function_name = tool_call.function.name
    function_to_call = available_functions[function_name]
    function_args = json.loads(tool_call.function.arguments)
    function_response = function_to_call(
        location=function_args.get("location"),
        unit=function_args.get("unit"),
    )

    # 関数の実行結果を会話履歴としてmessagesに追加
    messages.append(
        {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": function_response,
        }
    )

# 関数の実行結果もmessagesに含めた状態で、もう一度Chat Completions APIにリクエストを送る
second_response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
)
print(second_response.to_json(indent=2))
