from langsmith import Client

client = Client()
prompt = client.pull_prompt("zaki-yama/recipes")
prompt_value = prompt.invoke({"dish": "カレー"})
print(prompt_value)
