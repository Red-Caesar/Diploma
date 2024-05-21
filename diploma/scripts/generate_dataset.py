from diploma.generator import DatasetGenerator
import pandas as pd
import os

token = os.environ["OCTOAI_TOKEN"]
endpoint = os.environ["ENDPOINT"] + "/v1"
# model_name = "llama-2-70b-chat-fp16"
model_name = "mixtral-8x22b-finetuned"

generator = DatasetGenerator(model=model_name, token=token, endpoint=endpoint)
datasets = []
for i in range(1, 8):
    dataset = generator.generate_from_pdf(
        file_path=f"documents/Doc_#{i}.pdf",
        max_questions=100,
        seed=i,
        json_available=True,
    )
    datasets.append(dataset.to_pandas())

full_dataset = pd.concat(datasets, ignore_index=True)
full_dataset.to_csv("../datasets/synthetic_dataset_2.csv", index=False)
