import nest_asyncio
from langchain_community.document_loaders import GitLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)
documents = loader.load()


# Ragas による合成テストデータ生成
for document in documents:
    document.metadata["filename"] = document.metadata["source"]


nest_asyncio.apply()


generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings,
)
from ragas.testset.synthesizers import default_query_distribution

query_distribution = default_query_distribution(generator_llm)

testset = generator.generate_with_langchain_docs(
    documents,
    testset_size=4,
    query_distribution=query_distribution
)

# LangSmith の Dataset の作成
from langsmith import Client

dataset_name = "agent-book"

client = Client()

if client.has_dataset(dataset_name=dataset_name):
    client.delete_dataset(dataset_name=dataset_name)

dataset = client.create_dataset(dataset_name=dataset_name)


inputs = []
outputs = []
metadatas = []

for testset_record in testset.test_data:
    inputs.append(
        {
            "question": testset_record.question,
        }
    )
    outputs.append(
        {
            "contexts": testset_record.contexts,
            "ground_truth": testset_record.ground_truth,
        }
    )
    metadatas.append(
        {
            "source": testset_record.metadata[0]["source"],
            "evolution_type": testset_record.evolution_type,
        }
    )

client.create_examples(
    inputs=inputs,
    outputs=outputs,
    metadata=metadatas,
    dataset_id=dataset.id,
)
