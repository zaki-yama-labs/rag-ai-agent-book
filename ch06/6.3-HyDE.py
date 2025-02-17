from langchain_chroma import Chroma
from langchain_community.document_loaders import GitLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


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
print(len(documents))

# --- ドキュメントをベクトル化し、Chroma にインデクシングする
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(documents, embeddings)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

hypothetical_prompt = ChatPromptTemplate.from_template("""\
次の質問に回答する一文を書いてください。

質問: {question}
""")
hypothetical_chain = hypothetical_prompt | model | StrOutputParser()

prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
''')

retriever = db.as_retriever()

hyde_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "context": hypothetical_chain | retriever,
    }
    | prompt
    | model
    | StrOutputParser()
)

result = hyde_rag_chain.invoke("LangChainの概要を教えて")
print(result)
