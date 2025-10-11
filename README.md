# MedDRA-Coding-AI

English | [中文](README_zh.md)

<p align="center">
  <img src="images/Screenshot 2025-10-10 at 11.25.04 PM.png" alt="MedDRA RAG chat" width="280" />
  <img src="images/Screenshot 2025-10-10 at 11.27.31 PM.png" alt="Vector search sidebar" width="280" />
  <img src="images/Screenshot 2025-10-10 at 11.30.01 PM.png" alt="Index builder" width="280" />
</p>

MedDRA-Coding-AI is an open-source pharmacovigilance coding platform that automates MedDRA coding workflows with Retrieval-Augmented Generation (RAG). The assistant parses MedDRA ASCII dictionaries, builds semantic vector indexes, and offers a Streamlit UI that combines fast vector search with LLM reasoning across OpenAI, OpenRouter, or Ollama backends. Keywords: MedDRA coding automation, pharmacovigilance AI, RAG for safety narratives, adverse event classification, WHODrug roadmap.

## Features
- Automatic parsing of MedDRA ASCII.
- Unified term catalogue (`code`, `term`, `level`, parent relationships) for consistent embeddings.
- Hierarchy-aware documents: each LLT is expanded to include its PT/HLT/HLGT/SOC context before vectorisation, improving retrieval and LLM reasoning.
- Vector indexes per MedDRA language/version using SentenceTransformer embeddings (default: `BAAI/bge-m3`), with support for FAISS (in-memory) or Chroma (persistent LangChain store via `langchain-huggingface`/`langchain-chroma`).
- Retrieval + reasoning pipeline that prompts an LLM with top-k candidates to pick the best code and explain the choice.
- Streamlit UI with interactive conversation panel and sidebar showing top vector matches.
- Modular LLM client supporting OpenAI API or local Ollama models.
- CLI utility to batch-build indexes for every MedDRA version folder.

## Project Structure
```
meddra_rag_assistant/
├── build_index.py           # CLI to parse MedDRA folders and build indexes
├── config.yaml              # Global configuration (paths, embedding, LLM settings)
├── main.py                  # Streamlit entrypoint
├── modules/
│   ├── parser.py            # MedDRA ASCII parsing utilities
│   ├── vectorizer.py        # Embedding + FAISS index builder
│   ├── retriever.py         # Vector search helpers
│   ├── hierarchy.py         # Code → hierarchy resolver
│   ├── llm_client.py        # OpenAI/Ollama/OpenRouter abstraction
│   └── rag_agent.py         # End-to-end RAG orchestration
├── requirements.txt
└── README.md
```

Runtime directories:

- `dict/Meddra/<language_version>/`: raw MedDRA ASCII folders (e.g. `english_24.0`).
- `indexes/meddra__<language>_<version>/`: FAISS index (`index.faiss`), hierarchy-aware documents (`documents.csv`), and metadata per version.

## Setup
1. Ensure Python 3.10+
2. Install dependencies:
   ```bash
   pip install -r meddra_rag_assistant/requirements.txt
   ```
3. Place MedDRA ASCII folders under `dict/Meddra/` (default path, configurable).

### Configuration
Edit `meddra_rag_assistant/config.yaml` to adjust:
- `meddra_data_dir`: root folder containing MedDRA versions.
- `indexes_dir`: where FAISS indexes are stored.
- `embedding`: model name, batch size, cosine normalisation.
- `retrieval`: default `top_k` and optional score threshold (vector scores are re-ranked with lexical matching).
- `output`: UI toggles such as `include_hierarchy`.
- `vector_store`: choose between `faiss` (default) or `chroma`; configure devices/collection naming for Chroma.
- `llm`: backend (`openai`, `ollama`, or `openrouter`) and model-specific parameters.

For OpenAI, ensure `OPENAI_API_KEY` is set. For Ollama, run the local server and adjust `ollama.url` if needed.
For OpenRouter, either set `OPENROUTER_API_KEY` or place the key inside `config.llm.openrouter.api_key`; headers such as `HTTP-Referer` and `X-Title` can also be tweaked in `config.yaml`. A `.env` file (loaded automatically via `python-dotenv`) can hold secrets, for example:
```
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-...
```

To switch to a Chroma-backed vector store with HuggingFace embeddings, adjust the configuration:
```yaml
embedding:
  model_name: BAAI/bge-small-en
  batch_size: 64
  normalize: true
  device: auto

vector_store:
  backend: chroma
  chroma:
    collection_prefix: meddra
    device: auto  # use "cuda" when an NVIDIA GPU is available
```
Rebuild the indexes with `--force` to materialize the new store.

## Building Indexes
Scan every MedDRA version and build indexes:
```bash
python meddra_rag_assistant/build_index.py
```

Rebuild with `--force` whenever you change how embeddings are constructed (e.g. updates to parser display text or switching vector-store backends).

To rebuild a specific version:
```bash
python meddra_rag_assistant/build_index.py --versions english_24.0 --force
```

The command outputs per-version status and writes the FAISS index plus `terms.csv`/`metadata.json` under `indexes/`.

## Running the Web UI
Launch Streamlit:
```bash
streamlit run meddra_rag_assistant/main.py
```

UI capabilities:
- Enter a free-text medical term and choose the MedDRA version.
- View top-k vector candidates (sidebar) with similarity scores.
- Inspect the selected code, full hierarchy (LLT → PT → HLT → HLGT → SOC), and LLM reasoning.
- Low-confidence warnings appear if similarity scores fall below the configured threshold.

## Programmatic Use
Instantiate the agent directly for scripting or integration tests:
```python
from pathlib import Path
from meddra_rag_assistant.modules.rag_agent import MeddraRAGAgent

agent = MeddraRAGAgent(Path("meddra_rag_assistant/config.yaml"))
response = agent.run(term="left lung proliferative focus", version_key="english_24.0")
print(response.best_match, response.hierarchy)
```

## Roadmap & Ideas

- Add WHODrug coding support (parsing B3 datasets, building multilingual vector stores, harmonising with MedDRA results).
- Train domain-tuned embedding and reranking models so challenging narratives.
- Provide model evaluation harnesses covering recall@k and human-in-the-loop review workflows.

Looking for contributors! See `README_zh.md` for a Chinese overview.

## License
This project depends on MedDRA data files, which require appropriate licensing from the MedDRA MSSO. Ensure compliance before distributing dictionaries or embeddings derived from them.
