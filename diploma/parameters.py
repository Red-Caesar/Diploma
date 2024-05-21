import os
from typing import Union
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings, OctoAIEmbeddings
from ragas.metrics import (
    Faithfulness,
    ContextRecall,
    ContextPrecision,
    AnswerRelevancy,
)
from diploma.text_splitter import CharactersTextSplitter

SplitterType = Union[
    RecursiveCharacterTextSplitter,
    CharactersTextSplitter,
    TokenTextSplitter,
    SemanticChunker,
]
EmbeddingType = Union[HuggingFaceEmbeddings, OctoAIEmbeddings]
MetricType = Union[Faithfulness, ContextRecall, ContextPrecision, AnswerRelevancy]

CHUNK_SIZES = [100, 200, 400, 800, 1600]
CHUNK_NUMS = [1, 2, 3, 4]
SPLITTERS = {
    # Recursively splits text. This splitting is trying to keep related pieces of text next to each other.
    "related": RecursiveCharacterTextSplitter,
    # Splits text based on a user defined character.
    "character": CharactersTextSplitter,
    # Splits text on tokens. (using tiktoken)
    "token": TokenTextSplitter,
    # First splits on sentences. Then combines ones next to each other if they are semantically similar enough.
    "semantic": SemanticChunker,
}
EMBEDDINGS = {
    "huggingface": HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    "octoai": OctoAIEmbeddings(
        endpoint_url=(os.environ["ENDPOINT"] + "/v1/embeddings"),
        octoai_api_token=os.environ["TOKEN"],
    ),
}
METRICS = {
    "faithfulness": Faithfulness(),
    "context_recall": ContextRecall(),
    "context_precision": ContextPrecision(),
    "answer_relevancy": AnswerRelevancy(),
}
RAG_SYSTEM_PROMPT = """
Your primary role is to provide an answer to the question based on the given text.

When generating answers, you MUST follow these rules:
1. Your answers must be directly derived from the content within the provided text. Do not make up, hallucinate, or generate answers that are not explicitly supported by the given text.
2. If you don't not the answer, when say: I don't know.
3. Phrases like ’based on the provided context’, ’according to the context’, etc, ARE NOT ALLOWED to appear.
4. Do not just copy words from the context. Answer the question in your own words.
"""
