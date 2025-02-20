from typing import Any

from langchain_chroma import Chroma
from langchain_cohere import CohereRerank
from langchain_community.document_loaders import GitLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field


class QueryGenerationOutput(BaseModel):
    queries: list[str] = Field(..., description="検索クエリのリスト")


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


# --- LangChainの公式ドキュメントを読み込む
loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)
documents = loader.load()

# --- ドキュメントをベクトル化し、Chroma にインデクシングする
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(documents, embeddings)


model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

query_generation_prompt = ChatPromptTemplate.from_template("""\
質問に対してベクターデータベースから関連文書を検索するために、
3つの異なる検索クエリを生成してください。
距離ベースの類似性検索の限界を克服するために、
ユーザーの質問に対して複数の視点を提供することが目標です。

質問: {question}
""")

query_generation_chain = (
    query_generation_prompt
    | model.with_structured_output(QueryGenerationOutput)
    | (lambda x: x.queries)
)

hypothetical_prompt = ChatPromptTemplate.from_template("""\
次の質問に回答する一文をかいてください。

質問: {question}
""")
hypothetical_chain = hypothetical_prompt | model | StrOutputParser()

retriever = db.as_retriever()

prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
''')


def rerank(inp: dict[str, Any], top_n: int = 3) -> list[Document]:
    question = inp["question"]
    documents = inp["documents"]

    cohere_reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=top_n)
    return cohere_reranker.compress_documents(documents=documents, query=question)


rerank_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "documents": retriever,
    }
    # 1つ前の出力がrerank関数の引数inp: dict[str, Any]に渡される
    | RunnablePassthrough.assign(context=rerank)
    | prompt
    | model
    | StrOutputParser()
)

result = rerank_rag_chain.invoke("LangChainの概要を教えて")
print(result)
