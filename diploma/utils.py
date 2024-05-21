from tqdm import tqdm
from typing import List, Any, Dict, Union, Tuple
import os
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from diploma.parameters import (
    EmbeddingType,
    MetricType,
    SplitterType,
    RAG_SYSTEM_PROMPT,
    SPLITTERS,
    EMBEDDINGS,
)

from datasets import Dataset
import pandas as pd
from ragas import evaluate, RunConfig
import datetime
from openai import OpenAI
from langchain.llms.octoai_endpoint import OctoAIEndpoint


def prepare_vectordb(
    splitter: SplitterType,
    chunk_s: int,
    full_context: str,
    embedding_function: EmbeddingType,
) -> Any:
    print("Start text splitting")
    if splitter is SemanticChunker:
        text_splitter = splitter(
            embedding_function, breakpoint_threshold_type="percentile"
        )
    else:
        text_splitter = splitter(chunk_size=chunk_s, chunk_overlap=0)
    texts = text_splitter.split_text(full_context)

    print("Creating vector store")
    db = Chroma.from_texts(texts, embedding_function)
    return db


def run_RAG(
    model_name: str,
    vectordb: Any,
    num_relative_docs: int,
    questions: List[str],
) -> Tuple[List[str], List[list[str]], bool]:
    if os.environ["TOKEN"] is None or os.environ["ENDPOINT"] is None:
        raise ValueError("You should set the TOKEN and ENDPOINT environment variables.")

    client = OpenAI(
        api_key=os.environ["TOKEN"], base_url=os.environ["ENDPOINT"] + "/v1"
    )
    responses = []
    contexts = []

    is_token_limit_exceed = False
    for question in tqdm(questions, desc="Run RAG"):
        relative_docs = vectordb.similarity_search(question, k=num_relative_docs)
        relative_texts = [doc.page_content for doc in relative_docs]
        try:
            full_context = "".join(set(relative_texts))
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Question: {question}\n Context: {full_context}\n Your answer:",
                    },
                ],
                temperature=0.0,
                top_p=1.0,
                max_tokens=1000,
            )
            response_content = response.choices[0].message.content
        except Exception as e:
            print(f"RAG ERROR: {e}")
            response_content = "Dummy response"
            relative_texts = ["" for _ in range(num_relative_docs)]
            if "token count" in e.message:
                is_token_limit_exceed = True
            break
        responses.append(response_content)
        contexts.append(relative_texts)

    return responses, contexts, is_token_limit_exceed


def create_hg_dataset(
    benchmark: pd.DataFrame,
    contexts: List[List[str]],
    rag_answers: List[str],
) -> Dataset:
    df = pd.DataFrame(
        {
            "question": benchmark["questions"].to_list(),
            "contexts": contexts,
            "answer": rag_answers,
            "ground_truth": benchmark["answers"].to_list(),
        }
    )

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    folder_path = "diploma/results/RAG/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    df.to_csv(folder_path + timestamp + ".csv")
    return Dataset.from_pandas(df)


def ragas_eval(
    judge_name: str,
    ds: Dataset,
    metrics: List[MetricType],
) -> Dict[str, float]:
    if os.environ["TOKEN"] is None or os.environ["ENDPOINT"] is None:
        raise ValueError("You should set the TOKEN and ENDPOINT environment variables.")

    judge = OctoAIEndpoint(
        octoai_api_token=os.environ["TOKEN"],
        endpoint_url=os.environ["ENDPOINT"] + "/v1/chat/completions",
        model_kwargs={
            "model": judge_name,
            "messages": [],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 2500,
        },
    )
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Run RAGAS evaluation")
    run_config = RunConfig(max_workers=8)
    eval_result = evaluate(
        ds,
        metrics=metrics,
        llm=judge,
        embeddings=embedding_function,
        raise_exceptions=False,
        run_config=run_config,
    )
    return eval_result


def collect_test_results(evaluation_results: List[Dict[str, float]]) -> None:
    test_dict = {}
    for evaluation in evaluation_results:
        for key, val in evaluation.items():
            if key not in test_dict:
                test_dict[key] = [val]
            else:
                test_dict[key].append(val)
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    folder_path = "diploma/results/RAGAS/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    pd.DataFrame(test_dict).to_csv(
        folder_path + f"test_{timestamp}" + ".csv", index=False
    )


def collect_final_results(
    splitter: SplitterType,
    embedding_function: EmbeddingType,
    chunk_s: int,
    chunk_n: int,
    metric: Dict[str, float],
    final_df: Dict[str, List[Union[float, int, str]]],
) -> None:
    splitter_name = ""
    for key, val in SPLITTERS.items():
        if splitter is val:
            splitter_name = key
            break

    embedding_name = ""
    for key, val in EMBEDDINGS.items():
        if embedding_function is val:
            embedding_name = key
            break

    for key, val in metric.items():
        if key not in final_df:
            final_df[key] = [val]
        else:
            final_df[key].append(val)

    final_df["splitter"].append(splitter_name)
    final_df["embedding_function"].append(embedding_name)
    final_df["chunk_size"].append(chunk_s)
    final_df["chunk_num"].append(chunk_n)

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    folder_path = "diploma/results/RAGAS/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    pd.DataFrame(final_df).to_csv(folder_path + f"{timestamp}.csv")
