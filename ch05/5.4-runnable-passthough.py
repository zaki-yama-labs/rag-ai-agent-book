import pprint

from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
''')
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = TavilySearchAPIRetriever(k=3)

chain = {
    "context": retriever,
    "question": RunnablePassthrough(),
} | RunnablePassthrough.assign(answer=prompt | model | StrOutputParser())

output = chain.invoke("東京の今日の天気は？")
pprint.pprint(output)
