from typing import List
import pandas as pd

from diploma.parameters import EmbeddingType, MetricType, SplitterType
from diploma.utils import (
    prepare_vectordb,
    run_RAG,
    create_hg_dataset,
    ragas_eval,
    collect_test_results,
    collect_final_results,
)


def run_benchmark(
    benchmark: pd.DataFrame,
    full_context: str,
    chunk_sizes: List[int],
    chunk_nums: List[int],
    model_name: str,
    judge_name: str,
    embedding_functions: List[EmbeddingType],
    metrics: List[MetricType],
    splitters: List[SplitterType],
    repeat_ragas_eval: int = 1,
) -> pd.DataFrame:
    final_df = {
        "chunk_size": [],
        "chunk_num": [],
        "embedding_function": [],
        "splitter": [],
    }
    for splitter in splitters:
        for embedding_function in embedding_functions:
            for chunk_s in chunk_sizes:
                for chunk_n in chunk_nums:
                    db = prepare_vectordb(
                        splitter, chunk_s, full_context, embedding_function
                    )
                    try:
                        rag_results, relative_docs, is_token_limit_exceed = run_RAG(
                            model_name, db, chunk_n, benchmark["questions"].to_list()
                        )
                        db.delete_collection()
                        if is_token_limit_exceed:
                            break
                        ds = create_hg_dataset(benchmark, relative_docs, rag_results)
                        evaluation_results = []
                        for _ in range(repeat_ragas_eval):
                            ragas_result = ragas_eval(judge_name, ds, metrics)
                            evaluation_results.append(ragas_result)

                        if repeat_ragas_eval > 1:
                            collect_test_results(evaluation_results)

                        collect_final_results(
                            splitter,
                            embedding_function,
                            chunk_s,
                            chunk_n,
                            evaluation_results[-1],
                            final_df,
                        )
                    except Exception as e:
                        print(e)
    return pd.DataFrame(final_df)
