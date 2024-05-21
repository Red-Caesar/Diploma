# RAG evaluation system
## Install

```bash
git clone https://github.com/Red-Caesar/Diploma.git
git submodule update --init --recursive
```
## Prepare environment
If you don't have poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 - --version 1.7.1
```
One way to build this project is by using the following steps:
```bash
poetry config virtualenvs.in-project true
python3 -m venv .venv
source .venv/bin/activate
poetry install
```
To use LLM, you should set the TOKEN and ENDPOINT variables in your environment. One way to do this is:
1. Set the variables in the `diploma/.env_template` file;
2. Run `source diploma/.env_template`.

## Run evaluation
To run the evaluation, use the command:
```bash
python3 main.py
```
Also, you can customize the evaluation using flags:
```bash
--rag_model
--ragas_model
--limit
--chunk_sizes
--chunk_nums
--splitter
--embeddings
--metrics
--repeat
```
Another example of a command:
```bash
python3 main.py \
    --chunk_sizes 500 \
    --chunk_nums 2 \
    --splitter token \
    --embeddings octoai \
    --metrics faithfulness,context_recall \
    --limit 10
```
