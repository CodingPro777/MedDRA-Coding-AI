from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .hierarchy import MeddraHierarchy
from .llm_client import LLMClient
from .parser import MeddraParser, ParsedMeddraData
from .retriever import RetrievalResult, VectorIndex, VectorRetriever
from .utils import load_yaml, read_json
from .vectorstore_chroma import ChromaRetriever


SYSTEM_PROMPT = """You are a professional MedDRA coding expert.
Given a user-entered medical term and candidate MedDRA documents (each includes LLT/PT/HLT/HLGT/SOC context with MedDRA codes),
select the code that best matches the user input.
Only choose from the provided candidates.
Return a JSON object with fields:
  - best_match: { "term": string, "code": string, "level": string, "score": number }
  - confidence: float between 0 and 1
  - reason: short explanation of your choice
If no candidate is suitable, set best_match to null and explain what additional information is required.
"""


@dataclass
class RAGResponse:
    input_term: str
    selected_code: Optional[str]
    hierarchy: Dict[str, Optional[Dict[str, str]]]
    reasoning: str
    best_match: Optional[Dict[str, str]]
    candidates: List[RetrievalResult]
    llm_output: Optional[str]
    low_confidence: bool


class MeddraRAGAgent:
    """High-level orchestration for MedDRA Retrieval-Augmented Generation."""

    def __init__(self, config_path: Path):
        self.root_config = load_yaml(config_path)
        self.meddra_data_dir = Path(self.root_config.get("meddra_data_dir", "./dict/Meddra"))
        self.indexes_dir = Path(self.root_config.get("indexes_dir", "./indexes"))
        self.embedding_config = self.root_config.get("embedding", {})
        self.retrieval_config = self.root_config.get("retrieval", {})
        self.output_config = self.root_config.get("output", {})
        self.include_hierarchy = bool(self.output_config.get("include_hierarchy", True))
        self.vector_store_config = self.root_config.get("vector_store", {})
        self.vector_backend = self.vector_store_config.get("backend", "faiss").lower()
        self.llm_client = LLMClient.from_dict(self.root_config.get("llm", {}))

        self._parsed_cache: Dict[str, ParsedMeddraData] = {}
        self._hierarchy_cache: Dict[str, MeddraHierarchy] = {}
        self._index_cache: Dict[str, VectorIndex] = {}
        self._retriever_cache: Dict[str, object] = {}

    # ------------------------------------------------------------------ #
    # Version management
    # ------------------------------------------------------------------ #
    def available_versions(self) -> List[str]:
        versions: List[str] = []
        if not self.indexes_dir.exists():
            return versions
        for path in sorted(self.indexes_dir.glob("meddra__*")):
            if path.is_dir():
                versions.append(path.name.replace("meddra__", ""))
        return versions

    def ensure_resources(self, version_key: str) -> None:
        index_path = self.indexes_dir / f"meddra__{version_key}"
        if not index_path.exists():
            raise FileNotFoundError(f"Index for version '{version_key}' not found in {self.indexes_dir}.")

        metadata_path = index_path / "metadata.json"
        metadata: Dict[str, str] = {}
        if metadata_path.exists():
            metadata = read_json(metadata_path)

        backend = metadata.get("vector_store", self.vector_backend).lower()

        if backend == "faiss":
            if version_key not in self._index_cache:
                self._index_cache[version_key] = VectorIndex.load(index_path)
            if version_key not in self._retriever_cache:
                vector_index = self._index_cache[version_key]
                model_name = self.embedding_config.get("model_name", vector_index.model_name)
                normalize = self.embedding_config.get("normalize", True)
                batch_size = self.embedding_config.get("batch_size", 32)
                self._retriever_cache[version_key] = VectorRetriever(
                    vector_index=vector_index,
                    model_name=model_name,
                    normalize=normalize,
                    batch_size=batch_size,
                )
        elif backend == "chroma":
            if version_key not in self._retriever_cache:
                chroma_conf = self.vector_store_config.get("chroma", {})
                collection_name = metadata.get("collection_name") or f"meddra_{version_key.replace('.', '_')}"
                device_pref = chroma_conf.get("device", self.embedding_config.get("device", metadata.get("device", "auto")))
                model_name = self.embedding_config.get("model_name", metadata.get("model_name", "BAAI/bge-small-en"))
                documents_path = index_path / "documents.csv"
                self._retriever_cache[version_key] = ChromaRetriever(
                    index_dir=index_path,
                    model_name=model_name,
                    device=device_pref,
                    collection_name=collection_name,
                    documents_path=documents_path,
                    encode_batch_size=int(self.embedding_config.get("batch_size", 64)),
                )
        else:
            raise ValueError(f"Unsupported vector store backend: {backend}")

        if version_key not in self._parsed_cache:
            version_dir = self.meddra_data_dir / version_key
            parser = MeddraParser(version_dir)
            self._parsed_cache[version_key] = parser.parse()

        if self.include_hierarchy and version_key not in self._hierarchy_cache:
            parsed = self._parsed_cache[version_key]
            self._hierarchy_cache[version_key] = MeddraHierarchy(parsed)

    # ------------------------------------------------------------------ #
    # Core pipeline
    # ------------------------------------------------------------------ #
    def run(
        self,
        *,
        term: str,
        version_key: str,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        self.ensure_resources(version_key)
        retriever = self._retriever_cache[version_key]
        hierarchy = self._hierarchy_cache.get(version_key)

        top_k = top_k or int(self.retrieval_config.get("top_k", 5))
        results = retriever.search(term, top_k=top_k)
        score_threshold = float(self.retrieval_config.get("score_threshold", 0.0))
        low_confidence = not results or (score_threshold > 0 and results and results[0].score < score_threshold)

        empty_hierarchy = {"LLT": None, "PT": None, "HLT": None, "HLGT": None, "SOC": None} if self.include_hierarchy else {}

        if not results:
            reasoning = "No relevant MedDRA terms were retrieved. Please refine the input term or provide more context."
            return RAGResponse(
                input_term=term,
                selected_code=None,
                hierarchy=empty_hierarchy,
                reasoning=reasoning,
                best_match=None,
                candidates=[],
                llm_output=None,
                low_confidence=True,
            )

        prompt = self._build_prompt(term, results)
        llm_output = self.llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT)
        parsed_json = self._parse_llm_output(llm_output)

        selected_code, selected_level, reasoning = self._extract_selection(parsed_json, results)
        if not selected_code and results:
            selected_code = results[0].code
            selected_level = results[0].level
            reasoning = reasoning or "Defaulted to the top vector search result due to unparseable LLM output."

        best_match = parsed_json.get("best_match") if isinstance(parsed_json, dict) else None
        if selected_code:
            selected_result = next((r for r in results if r.code == selected_code), None)
            if selected_result:
                if not best_match:
                    best_match = {
                        "term": selected_result.term,
                        "code": selected_result.code,
                        "level": selected_result.level,
                        "score": selected_result.score,
                        "document_text": selected_result.document_text,
                    }
                else:
                    best_match.setdefault("term", selected_result.term)
                    best_match.setdefault("level", selected_result.level)
                    best_match.setdefault("score", selected_result.score)
                    best_match.setdefault("document_text", selected_result.document_text)

        hierarchy_dict = empty_hierarchy
        if self.include_hierarchy and hierarchy:
            hierarchy_item = (
                hierarchy.resolve(selected_code, selected_level)
                if selected_code
                else hierarchy.resolve(results[0].code, results[0].level)
            )
            hierarchy_dict = hierarchy_item.to_dict()

        return RAGResponse(
            input_term=term,
            selected_code=selected_code,
            hierarchy=hierarchy_dict,
            reasoning=reasoning or llm_output,
            best_match=best_match,
            candidates=results,
            llm_output=llm_output,
            low_confidence=low_confidence,
        )

    # ------------------------------------------------------------------ #
    # Prompting helpers
    # ------------------------------------------------------------------ #
    def _build_prompt(self, term: str, candidates: List[RetrievalResult]) -> str:
        candidate_lines = []
        for index, item in enumerate(candidates, start=1):
            doc_text = item.document_text.strip()
            if not doc_text:
                doc_text = item.metadata.get("display_text", "")
            hierarchy_snippet = doc_text
            candidate_lines.append(
                f"{index}. Code={item.code}, Term={item.term}, VectorScore={item.score:.3f}, CombinedScore={item.combined_score:.3f}\n{hierarchy_snippet}"
            )

        prompt = (
            f'User term: "{term}"\n\n'
            "Candidate documents:\n"
            f"{'\n\n'.join(candidate_lines)}\n\n"
            "Respond with strictly valid JSON."
        )
        return prompt

    def _parse_llm_output(self, llm_output: str) -> Dict:
        llm_output = llm_output.strip()
        if not llm_output:
            return {}
        try:
            return json.loads(llm_output)
        except json.JSONDecodeError:
            start = llm_output.find("{")
            end = llm_output.rfind("}")
            if start != -1 and end != -1 and end > start:
                fragment = llm_output[start : end + 1]
                try:
                    return json.loads(fragment)
                except json.JSONDecodeError:
                    pass
        return {}

    def _extract_selection(
        self,
        parsed_json: Dict,
        candidates: List[RetrievalResult],
    ) -> Tuple[Optional[str], Optional[str], str]:
        if not isinstance(parsed_json, dict):
            return None, None, ""
        best_match = parsed_json.get("best_match") or {}
        selected_code = best_match.get("code")
        selected_level = best_match.get("level")

        if selected_code:
            for result in candidates:
                if str(result.code) == str(selected_code):
                    selected_code = result.code
                    selected_level = result.level
                    break

        reasoning = parsed_json.get("reason") or parsed_json.get("reasoning") or ""
        return selected_code, selected_level, reasoning
