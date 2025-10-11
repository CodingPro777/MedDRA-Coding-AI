from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

from .utils import dump_json


@dataclass
class IndexMetadata:
    model_name: str
    normalize: bool
    dimension: int
    language: str
    version: str
    terms_count: int
    vector_store: str = "faiss"

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "normalize": self.normalize,
            "dimension": self.dimension,
            "language": self.language,
            "version": self.version,
            "terms_count": self.terms_count,
            "vector_store": self.vector_store,
        }


class IndexBuilder:
    """Vectorise MedDRA terms and persist them to a FAISS index."""

    def __init__(self, model_name: str, *, batch_size: int = 64, normalize: bool = True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build(
        self,
        *,
        documents: pd.DataFrame,
        output_dir: Path,
        language: str,
        version: str,
    ) -> IndexMetadata:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        documents = documents.copy()
        documents = documents.reset_index(drop=True)
        documents.to_csv(output_dir / "documents.csv", index=False)

        if "document_text" not in documents.columns:
            raise ValueError("documents DataFrame must contain a 'document_text' column")

        texts = documents["document_text"].tolist()
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=self.normalize,
        )
        embeddings = embeddings.astype("float32")

        dimension = embeddings.shape[1]
        if self.normalize:
            index = faiss.IndexFlatIP(dimension)
        else:
            index = faiss.IndexFlatL2(dimension)

        index.add(embeddings)
        faiss.write_index(index, str(output_dir / "index.faiss"))

        metadata = IndexMetadata(
            model_name=self.model_name,
            normalize=self.normalize,
            dimension=dimension,
            language=language,
            version=version,
            terms_count=len(documents),
            vector_store="faiss",
        )
        dump_json(output_dir / "metadata.json", metadata.to_dict())
        return metadata
