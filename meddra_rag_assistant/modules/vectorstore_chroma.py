from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from langchain.docstore.document import Document

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
    except ImportError:
        from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore

try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma  # type: ignore
    except ImportError:
        from langchain.vectorstores import Chroma  # type: ignore
from rapidfuzz import fuzz

from .retriever import RetrievalResult
from .utils import dump_json


def resolve_device(preferred: Optional[str]) -> str:
    if preferred and preferred.lower() not in {"", "auto"}:
        return preferred
    try:
        import torch  # noqa: F401
    except ImportError:
        return "cpu"
    else:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"


class ChromaIndexBuilder:
    """Build a Chroma vector store populated with hierarchy-aware documents."""

    def __init__(
        self,
        model_name: str,
        *,
        device: Optional[str] = None,
        collection_prefix: str = "meddra",
        encode_batch_size: Optional[int] = None,
        add_batch_size: int = 2048,
    ):
        self.model_name = model_name
        self.device = resolve_device(device)
        self.collection_prefix = collection_prefix
        self.encode_batch_size = encode_batch_size
        self.add_batch_size = max(1, add_batch_size)
        self._embedding: Optional[HuggingFaceEmbeddings] = None

    @property
    def embedding(self) -> HuggingFaceEmbeddings:
        if self._embedding is None:
            encode_kwargs = {}
            if self.encode_batch_size:
                encode_kwargs["batch_size"] = self.encode_batch_size
            self._embedding = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": self.device},
                encode_kwargs=encode_kwargs,
            )
        return self._embedding

    def _collection_name(self, language: str, version: str) -> str:
        suffix = f"{language}_{version}".replace(".", "_")
        return f"{self.collection_prefix}_{suffix}"

    def build(
        self,
        *,
        documents: pd.DataFrame,
        output_dir: Path,
        language: str,
        version: str,
    ) -> Dict[str, object]:
        output_dir = Path(output_dir)
        if output_dir.exists():
            for item in output_dir.iterdir():
                if item.is_file():
                    item.unlink()
                else:
                    shutil.rmtree(item)
        output_dir.mkdir(parents=True, exist_ok=True)

        documents = documents.copy().fillna("")
        documents.to_csv(output_dir / "documents.csv", index=False)

        if documents.shape[0] == 0:
            metadata = {
                "vector_store": "chroma",
                "model_name": self.model_name,
                "language": language,
                "version": version,
                "device": self.device,
                "collection_name": self._collection_name(language, version),
                "documents_count": 0,
            }
            dump_json(output_dir / "metadata.json", metadata)
            return metadata

        collection_name = self._collection_name(language, version)
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding,
            persist_directory=str(output_dir),
            collection_metadata={"hnsw:construction_ef": 200},
        )
        docs: List[Document] = []
        for _, row in documents.iterrows():
            metadata = {key: (value if isinstance(value, str) else str(value)) for key, value in row.items()}
            page_content = metadata.get("document_text") or metadata.get("display_text") or metadata.get("term") or ""
            if not page_content:
                continue
            docs.append(Document(page_content=page_content, metadata=metadata))

        if not docs:
            metadata = {
                "vector_store": "chroma",
                "model_name": self.model_name,
                "language": language,
                "version": version,
                "device": self.device,
                "collection_name": collection_name,
                "documents_count": 0,
            }
            dump_json(output_dir / "metadata.json", metadata)
            return metadata

        batch_size = max(1, min(self.add_batch_size, 4096))
        for start in range(0, len(docs), batch_size):
            vectorstore.add_documents(docs[start : start + batch_size])
        if hasattr(vectorstore, "persist"):
            vectorstore.persist()

        metadata = {
            "vector_store": "chroma",
            "model_name": self.model_name,
            "language": language,
            "version": version,
            "device": self.device,
            "collection_name": collection_name,
            "documents_count": len(docs),
        }
        dump_json(output_dir / "metadata.json", metadata)
        return metadata


class ChromaRetriever:
    """Perform similarity search against a persisted Chroma collection."""

    def __init__(
        self,
        *,
        index_dir: Path,
        model_name: str,
        device: Optional[str],
        collection_name: str,
        documents_path: Path,
        encode_batch_size: Optional[int] = None,
    ):
        self.index_dir = Path(index_dir)
        self.device = resolve_device(device)
        encode_kwargs = {}
        if encode_batch_size:
            encode_kwargs["batch_size"] = encode_batch_size
        self.embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=encode_kwargs,
        )
        self.collection_name = collection_name
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding,
            persist_directory=str(self.index_dir),
        )
        if documents_path.exists():
            self.documents = pd.read_csv(documents_path, dtype=str).fillna("")
        else:
            self.documents = pd.DataFrame(columns=["term"])

    def _exact_match_results(
        self,
        query_lower: str,
        seen_codes: set[str],
        top_k: int,
    ) -> List[RetrievalResult]:
        matches = self.documents[self.documents["term"].str.lower() == query_lower]
        results: List[RetrievalResult] = []
        remaining = max(0, top_k - len(seen_codes))
        if remaining <= 0:
            return results
        for _, row in matches.head(remaining).iterrows():
            code = str(row.get("llt_code") or row.get("doc_id") or row.get("code", ""))
            if code in seen_codes:
                continue
            metadata = row.to_dict()
            result = RetrievalResult(
                code=code,
                term=str(row.get("term") or row.get("llt_term") or ""),
                level=str(row.get("level", "")),
                score=0.0,
                metadata=metadata,
                lexical_score=1.0,
                combined_score=1.1,
                document_text=str(row.get("document_text", "")),
            )
            results.append(result)
            seen_codes.add(code)
        return results

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if not query.strip():
            return []

        query_lower = query.strip().lower()
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
        results: List[RetrievalResult] = []
        seen_codes: set[str] = set()

        for doc, distance in docs_with_scores:
            metadata = dict(doc.metadata)
            code = str(metadata.get("llt_code") or metadata.get("doc_id") or metadata.get("code", ""))
            term = str(metadata.get("term") or metadata.get("llt_term") or "").strip()
            level = str(metadata.get("level", "") or "LLT")
            result = RetrievalResult(
                code=code,
                term=term,
                level=level,
                score=float(distance),
                metadata=metadata,
                document_text=doc.page_content,
            )
            results.append(result)
            seen_codes.add(code)

        if query_lower:
            results.extend(self._exact_match_results(query_lower, seen_codes, top_k))

        for result in results:
            lexical_source = " ".join(filter(None, [result.term, result.document_text]))
            lexical = fuzz.token_set_ratio(query_lower, lexical_source.lower()) if lexical_source else 0
            lexical_norm = lexical / 100.0
            vector_norm = 1.0 / (1.0 + max(result.score, 0.0))
            combined = 0.7 * vector_norm + 0.3 * lexical_norm
            if lexical >= 95:
                combined += 0.3
            result.lexical_score = lexical_norm
            result.combined_score = combined

        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results[:top_k]
