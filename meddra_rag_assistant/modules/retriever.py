from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

import faiss
import pandas as pd
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

from .utils import read_json


@dataclass
class RetrievalResult:
    code: str
    term: str
    level: str
    score: float
    metadata: dict
    lexical_score: float = 0.0
    combined_score: float = 0.0
    document_text: str = ""


class VectorIndex:
    """Wrapper around a persisted FAISS index and its metadata."""

    def __init__(self, *, index: faiss.Index, metadata: dict, terms: pd.DataFrame):
        self.index = index
        self.metadata = metadata
        self.terms = terms

    @classmethod
    def load(cls, index_dir: Path) -> "VectorIndex":
        index_dir = Path(index_dir)
        metadata = read_json(index_dir / "metadata.json")
        documents_path = index_dir / "documents.csv"
        terms_path = index_dir / "terms.csv"
        if documents_path.exists():
            terms = pd.read_csv(documents_path, dtype=str).fillna("")
        elif terms_path.exists():
            terms = pd.read_csv(terms_path, dtype=str).fillna("")
        else:
            raise FileNotFoundError(f"No documents.csv or terms.csv found in {index_dir}")
        index = faiss.read_index(str(index_dir / "index.faiss"))
        return cls(index=index, metadata=metadata, terms=terms)

    @property
    def normalize(self) -> bool:
        return bool(self.metadata.get("normalize", True))

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("model_name"))


class VectorRetriever:
    """Perform vector similarity search over a MedDRA FAISS index."""

    def __init__(
        self,
        *,
        vector_index: VectorIndex,
        model_name: Optional[str] = None,
        batch_size: int = 32,
        normalize: Optional[bool] = None,
    ):
        self.vector_index = vector_index
        self.model_name = model_name or vector_index.model_name
        self.batch_size = batch_size
        self.normalize = vector_index.normalize if normalize is None else normalize
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if not query.strip():
            return []

        embedding = self.model.encode(
            [query],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        ).astype("float32")

        distances, indices = self.vector_index.index.search(embedding, top_k)
        distances = distances[0]
        indices = indices[0]
        print(query, indices)
        query_lower = query.strip().lower()
        results: List[RetrievalResult] = []
        for idx, score in zip(indices, distances):
            if idx < 0 or idx >= len(self.vector_index.terms):
                continue
            term_row = self.vector_index.terms.iloc[idx].to_dict()
            code = str(
                term_row.get("llt_code")
                or term_row.get("doc_id")
                or term_row.get("code", "")
            )
            term_value = str(term_row.get("term") or term_row.get("llt_term") or "")
            level_value = str(term_row.get("level", ""))
            results.append(
                RetrievalResult(
                    code=code,
                    term=term_value,
                    level=level_value,
                    score=float(score),
                    metadata=term_row,
                    document_text=str(term_row.get("document_text", "")),
                )
            )

        # Exact term matches get appended even if not surfaced by the vector search.
        seen_codes: Set[str] = {r.code for r in results}
        if query_lower and "term" in self.vector_index.terms.columns:
            matches = self.vector_index.terms[
                self.vector_index.terms["term"].str.lower() == query_lower
            ]
            for _, row in matches.head(max(0, top_k - len(results))).iterrows():
                code = str(
                    row.get("llt_code")
                    or row.get("doc_id")
                    or row.get("code", "")
                )
                if code in seen_codes:
                    continue
                result = RetrievalResult(
                    code=code,
                    term=str(row.get("term") or row.get("llt_term") or ""),
                    level=str(row.get("level", "")),
                    score=1.0,
                    metadata=row.to_dict(),
                    lexical_score=1.0,
                    combined_score=1.1,
                    document_text=str(row.get("document_text", "")),
                )
                results.append(result)
                seen_codes.add(code)

        # Re-rank results using lexical similarity as a tie-breaker.
        for result in results:
            lexical_source = " ".join(
                filter(
                    None,
                    [
                        result.term,
                        result.document_text,
                    ],
                )
            )
            lexical = (
                fuzz.token_set_ratio(query_lower, lexical_source.lower())
                if lexical_source
                else 0
            )
            lexical_norm = lexical / 100.0
            vector_norm = max(min((result.score + 1.0) / 2.0, 1.0), 0.0)
            combined = 0.7 * vector_norm + 0.3 * lexical_norm
            if lexical >= 95:
                combined += 0.3
            result.lexical_score = lexical_norm
            result.combined_score = combined

        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results[:top_k]
