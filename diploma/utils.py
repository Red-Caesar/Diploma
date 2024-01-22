import json
from typing import List, Dict

def get_contexts(path: str) -> List[str]:
    with open(path, "r") as read_file:
        data = json.load(read_file)["data"]
    contexts = []
    for task in data:
        is_impossible = task["paragraphs"][0]["qas"][0]["is_impossible"]
        if not is_impossible: 
            contexts.append(task["paragraphs"][0]["context"])

    return contexts

def get_qa(path: str, limit: int=None) -> List[Dict[str, str]]:
    with open(path, "r") as read_file:
        data = json.load(read_file)["data"]
    qa = []

    for task in data:
        is_impossible = task["paragraphs"][0]["qas"][0]["is_impossible"]
        if not is_impossible:
            question = task["paragraphs"][0]["qas"][0]["question"]
            answer = task["paragraphs"][0]["qas"][0]["answers"][0]["text"]
            qa.append({"question": question, "answer": answer})

    return qa
