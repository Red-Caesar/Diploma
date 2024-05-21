from typing import List, Optional, Union
import random

from langchain_community.document_loaders import PyPDFLoader
from instructor import patch
from langchain.text_splitter import TokenTextSplitter
from openai import OpenAI
from tqdm import tqdm
import pandas as pd

from pydantic import BaseModel


class DatasetItem(BaseModel):
    question: str
    answer: str
    context: str


class Dataset(BaseModel):
    items: List[DatasetItem]

    def to_pandas(
        self,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "questions": [item.question for item in self.items],
                "answers": [item.answer for item in self.items],
                "contexts": [item.context for item in self.items],
            }
        )


_DEFAULT_PROMPT = """
You are an expert at understanding and analyzing documents.
Your primary role is to generate question and ground truth answer pairs based on the provided text.

When generating questions and answers, you MUST follow these rules:
1. Your ground truth answers must be directly derived from the content within the provided text. Do not make up, hallucinate, or generate answers that are not explicitly supported by the given text.
2. The question should be fully answerable from information present in given context.
3. Make sure the question is clear and unambiguous.
4. Phrases like ’based on the provided context’, ’according to the context’, etc, are not allowed to appear inthe question.
5. Include the relevant 'context' paragraph from which you generated each question and ground truth answer pair. The 'context' paragraph MUST contain the specific information that supports the ground truth answer.
6. If the provided text does not contain sufficient information to generate a question-answer pair, do not attempt to create one.
7. The answer must use the information provided in the context.
8. Do not just copy words from the context. Answer the question in your own words.
9. Your responses should be in the following format:
   Question: [Generated question]
   Answer: [Ground truth answer]
   Context: [Relevant paragraph from the text that supports the answer]

Remember, your primary objective is to create accurate, grounded, and contextually relevant question-answer pairs while strictly avoiding any fabrication or speculation.
"""


def _get_json_prompt(json_schema: str) -> str:
    return f"""
Generate a JSON object that conforms to the following JSON schema:
<JSON_SCHEMA>{ json_schema }</JSON_SCHEMA>
**Constraints:**
* NEVER include any introductory text.
"""


class DatasetGenerator:
    def __init__(self, model: str, token: str, endpoint: str):
        if not token:
            raise ValueError("API key is required.")

        self._model = model
        self._client = patch(OpenAI(api_key=token, base_url=endpoint))

    def generate_from_texts(
        self,
        texts: List[str],
        max_questions=10,
        json_available: bool = False,
        **kwargs,
    ) -> Dataset:
        system_prompt = kwargs.get("system_prompt", _DEFAULT_PROMPT)
        max_tokens = kwargs.get("max_tokens", 2000)

        num_texts = len(texts)
        questions_per_text = max_questions // num_texts
        num_remaining_questions = max_questions % num_texts

        progress_bar = tqdm(total=max_questions, desc="Generating questions")

        items: List[DatasetItem] = []
        for index, text in enumerate(texts):
            try:
                current_max_questions = questions_per_text
                if index < num_remaining_questions:
                    current_max_questions += 1

                if json_available:
                    response_format = {
                        "type": "json_object",
                        "schema": Dataset.model_json_schema(),
                    }
                    response = self._client.chat.completions.create(
                        model=self._model,
                        response_format=response_format,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": f"Generate {current_max_questions} questions for the following block of text: {text}",
                            },
                        ],
                        max_tokens=max_tokens,
                    )
                else:
                    response = self._client.chat.completions.create(
                        model=self._model,
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt
                                + _get_json_prompt(Dataset.model_json_schema()),
                            },
                            {
                                "role": "user",
                                "content": f"Generate {current_max_questions} questions for the following block of text: {text}. JSON:",
                            },
                        ],
                        max_tokens=max_tokens,
                    )
                jsonstr = response.choices[0].message.content
                response = Dataset.model_validate_json(jsonstr, strict=True)
                items.extend(response.items)

                progress_bar.update(len(response.items))

                if len(items) >= max_questions:
                    break

            except Exception as e:
                print(f"Failed to generate questions for batch {index + 1}: {e}")
                continue
        progress_bar.close()

        return Dataset(
            items=items[:max_questions],
        )

    def generate_from_pdf(
        self,
        file_path: str,
        max_questions: int = 10,
        samples_size: Optional[Union[float, int]] = None,
        seed: int = 2024,
        json_available: bool = False,
        **kwargs,
    ) -> Dataset:
        chunk_size = kwargs.get("chunk_size", 1024)
        chunk_overlap = kwargs.get("chunk_overlap", 128)

        token_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        documents_pages = [doc.page_content.replace("\n", " ") for doc in documents]
        if samples_size:
            random.seed(seed)
            if isinstance(samples_size, float):
                samples_size = int(len(documents_pages) * samples_size)
            start_index = random.randint(0, len(documents_pages) - samples_size)
            documents_pages = documents_pages[start_index : start_index + samples_size]

        text = "\n".join(documents_pages)
        texts = token_splitter.split_text(text)

        return self.generate_from_texts(
            texts=texts,
            max_questions=max_questions,
            json_available=json_available,
            kwargs=kwargs,
        )
