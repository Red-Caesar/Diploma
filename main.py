import argparse
import pandas as pd
import os
import datetime

from diploma import evaluate
from diploma.parameters import (
    CHUNK_SIZES,
    CHUNK_NUMS,
    SPLITTERS,
    EMBEDDINGS,
    METRICS,
)

from langchain_community.document_loaders import PyPDFLoader


# "mistral-7b-instruct-fp16"
# "llama-2-70b-chat-fp16"
# "mixtral-8x7b-instruct"
# "meta-llama-3-70b-instruct"
def parse_args():
    """
    :param rag_model: str
        The name of the model for RAG pipeline.
    :param ragas_model: str
        The name of the model is used to evaluate the RAG pipeline.
    :param limit: float
        Limit the number of examples per task (only use this for testing).
        If <1, limit is a percentage of the total number of examples.
    :param chunk_sizes: int
        The size of one chunk (can be interpreted differently by different splitters).
    :param chunk_nums: int
        The number of retrieved chunks in a relevant search.
    :param splitter: Union[related, character, token, semantic]
        The method of dividing text into chunks.
    :param embeddings: Union[huggingface, octoai]
        The emdedding function in a relevant search.
    :param metrics: Union[faithfulness, context_recall, context_precision, answer_relevancy]
        Metrics to evaluate RAG performance.
    :param repeat: int
        Repeat the evaluation on the same RAG dataset (only use this for testing).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag_model", type=str, default="mistral-7b-instruct")
    parser.add_argument("--ragas_model", type=str, default="meta-llama-3-70b-instruct")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--chunk_sizes", type=str, default=None)
    parser.add_argument("--chunk_nums", type=str, default=None)
    parser.add_argument("--splitter", type=str, default=None)
    parser.add_argument("--embeddings", type=str, default=None)
    parser.add_argument("--metrics", type=str, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--path_to_context_file", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if os.environ["TOKEN"] is None or os.environ["ENDPOINT"] is None:
        raise ValueError("You should set the TOKEN and ENDPOINT environment variables.")

    if not args.chunk_sizes:
        chunk_sizes = CHUNK_SIZES
    else:
        chunk_sizes = list(map(int, args.chunk_sizes.split(",")))

    if not args.chunk_nums:
        chunk_nums = CHUNK_NUMS
    else:
        chunk_nums = list(map(int, args.chunk_nums.split(",")))

    if not args.splitter:
        splitters = list(SPLITTERS.values())
    else:
        splitters = args.splitter.split(",")
        try:
            splitters = [SPLITTERS[splitter] for splitter in splitters]
        except KeyError:
            raise KeyError(f"Available names for splitter: {list(SPLITTERS.keys())}")

    if not args.embeddings:
        embedding_functions = list(EMBEDDINGS.values())
    else:
        if args.embeddings not in EMBEDDINGS:
            raise ValueError(
                f"Available names for embedding_function: {list(EMBEDDINGS.keys())}"
            )
        embedding_functions = [EMBEDDINGS[args.embeddings]]

    if not args.metrics:
        metrics = list(METRICS.values())
    else:
        metrics = args.metrics.split(",")
        try:
            metrics = [METRICS[metric] for metric in metrics]
        except KeyError:
            raise KeyError(
                f"Available names for metrics: {list(METRICS.keys())}. Your names: {metrics}"
            )

    if args.repeat > 1:
        print(
            "WARNING. The repeat parameter is only needed for testing purposes and should not be used in other cases."
        )

    dataset = pd.read_csv("diploma/datasets/synthetic_dataset.csv")
    if args.limit is not None:
        limit = int(len(dataset) * args.limit) if args.limit < 1.0 else int(args.limit)
        dataset = dataset.head(limit)

    if args.path_to_context_file is None:
        documents_pages = []
        for i in range(1, 8):
            loader = PyPDFLoader(f"diploma/scripts/documents/Doc_#{i}.pdf")
            documents = loader.load()
            documents_pages += [
                doc.page_content.replace("\n", " ") for doc in documents
            ]
        full_context = "\n".join(documents_pages)
    else:
        with open(args.path_to_context_file, "r") as f:
            full_context = f.read()

    result = evaluate.run_benchmark(
        dataset,
        full_context,
        chunk_sizes,
        chunk_nums,
        args.rag_model,
        args.ragas_model,
        embedding_functions,
        metrics,
        splitters,
        args.repeat,
    )

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    folder_path = "diploma/results/RAGAS/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    result.to_csv(folder_path + f"final_result_{timestamp}.csv")


if __name__ == "__main__":
    main()
