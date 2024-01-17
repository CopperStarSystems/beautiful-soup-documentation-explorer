# beautiful-soup-documentation-explorer

Although BS4 provides comprehensive documentation, it's in a single page with over 100k characters, which makes it really
hard to cross-reference things without miles of scrolling.

Inspired by the [chat-langchain](https://chat.langchain.com/) project, I decided to create a RAG application that ingests the BS4 documentation
and allows it to be searched in a conversational manner.

## Installation

Install the LangChain CLI if you haven't yet

```shell
pip install -U langchain-cli
```

Install Dependencies
```shell
poetry install
```

## Configure Environment Variables
```shell
export OPENAI_API_KEY=...
```

## Ingest BS4 Documentation (One-time setup)
**Important:** Run this command from the root folder
```shell
./ingest.sh
```

## Launch LangServe
**Important:** Run this command from the root folder
```bash
langchain serve
```

# Stack
- `langchain-cli` starting from the `rag_chroma` template
- `python 3.11`
- `langchain`
- `chromadb` with local DB