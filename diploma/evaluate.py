from diploma.models.octoai_llms import OctoAIEndpointLM

from typing import List, Any, Dict
import os
import uuid
import pandas as pd
from tqdm import tqdm

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms.octoai_endpoint import OctoAIEndpoint

from datasets import Dataset
from ragas import evaluate


def run_RAG(
    model_name: str,
    vectordb: Any,
    num_relative_docs: int,
    tasks: List[str],
) -> List[str]:
    print("Run RAG")
    model = OctoAIEndpointLM(model_name=model_name)
    responses = []
    for task in tqdm(tasks):
        relative_docs = vectordb.similarity_search(task, k=num_relative_docs)
        full_context = [doc.page_content for doc in relative_docs]
        context = ''.join(set(full_context))
        message = f"Question: {task}\n Context: {context}"

        response = model.model_generate(message)
        responses.append(response)
    return responses


def create_hg_dataset(
    tasks: List[str],
    contexts: List[str],
    answers: List[str],
    ground_truths: List[str]
) -> Dataset:
    df = pd.DataFrame({
        "question": tasks,
        "contexts": [[el] for el in contexts],
        "answer": answers,
        "ground_truth": [el for el in ground_truths]
    })

    return Dataset.from_pandas(df)


def ragas_eval(
    judge_name: str,
    ds: Dataset,
    metrics: List[Any],
    embedding_function: Any,
    repeat_ragas_eval: int=1,
) -> List[Dict[str, float]]:

    token = os.environ["OCTOAI_TOKEN"]
    endpoint = os.environ["ENDPOINT"]

    if token is None:
            raise ValueError("TOKEN not found.")
    if endpoint is None:
            raise ValueError("ENDPOINT not found.")

    judge = OctoAIEndpoint(
        octoai_api_token=token,
        endpoint_url=endpoint + "/v1/chat/completions",
        model_kwargs={
            "model": judge_name,
            "messages": [],
            "temperature": 0.0,
            "top_p": 1.0,
        },
    )

    print("Run RAGAS evaluation")
    RAG_metrics = []
    for _ in range(repeat_ragas_eval):
        eval_res = evaluate(
        ds,
        metrics=metrics,
        llm=judge,
        embeddings=embedding_function,
        raise_exceptions=False,
        )
        RAG_metrics.append(eval_res)
    return RAG_metrics

def collect_results(
    model_name: str,
    chunk_s: int,
    chunk_n: int,
    metrics: List[Dict[str, float]],
    judge_name: str,
) -> Dict[str, Any]:
    res_dict = {"model_name": [], "chunk_size": [], "chunk_num": [], "judge": []}
    for metric in metrics:
        for key, val in metric.items():
            if key not in res_dict:
                res_dict[key] = [val]
            else:
                res_dict[key].append(val)

        res_dict["model_name"].append(model_name)
        res_dict["chunk_size"].append(chunk_s)
        res_dict["chunk_num"].append(chunk_n)
        res_dict["judge"].append(judge_name)
        print("Current results:", model_name, chunk_s, chunk_n, judge_name, metric)
    return res_dict


def prepare_vectordb(
    splitter: Any,
    chunk_s: int,
    documents: List[Document],
    embedding_function: Any,
) -> Any:
    print("Start text splitting")
    text_splitter = splitter(chunk_size=chunk_s, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    db = Chroma.from_documents(docs, embedding_function)
    return db


def run_benchmark(
    tasks: List[str],
    contexts: List[str],
    ground_truths: List[str],
    chunk_sizes: List[int],
    chunk_nums: List[int],
    models: List[str],
    judges: List[str],
    embedding_function: Any,
    metrics: List[Any],
    splitter: Any=RecursiveCharacterTextSplitter,
    repeat_ragas_eval: int=1,
) -> pd.DataFrame:
    doc_ids = [str(uuid.uuid4()) for _ in contexts]
    documents = [Document(page_content=s, metadata={"doc_id": doc_ids[i]}) for i, s in enumerate(contexts)]
    RAG_metrics = {"model_name": [], "chunk_size": [], "chunk_num": [], "judge": []}

    for model_name in models:
        for chunk_s in chunk_sizes:
            for chunk_n in chunk_nums:
                db = prepare_vectordb(splitter, chunk_s, documents, embedding_function)
                try:
                    results = run_RAG(model_name, db, chunk_n, tasks)
                    db.delete_collection()
                    
                    ds = create_hg_dataset(tasks, contexts, results, ground_truths)
                        
                    for judge_name in judges:
                        ragas_res = ragas_eval(judge_name, ds, metrics, embedding_function, repeat_ragas_eval)
                        evaluation_results = collect_results(model_name, chunk_s, chunk_n, ragas_res, judge_name)
                        for key, val in evaluation_results.items():
                            if key not in RAG_metrics:
                                RAG_metrics[key] = val
                            else:
                                RAG_metrics[key] += val
                except Exception as e:
                    print(e)
    return pd.DataFrame(RAG_metrics)
