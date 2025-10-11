#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from modules.parser import MeddraParser
from modules.vectorizer import IndexBuilder
from modules.utils import load_yaml


def discover_versions(root: Path) -> list[str]:
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FAISS indexes for MedDRA dictionaries.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to configuration YAML.")
    parser.add_argument("--data-dir", type=Path, help="Override MedDRA data directory.")
    parser.add_argument("--indexes-dir", type=Path, help="Override output directory for indexes.")
    parser.add_argument(
        "--versions",
        nargs="*",
        help="Specific MedDRA versions to build (e.g. english_24.0). Defaults to every directory in data dir.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild indexes even if they already exist.")
    args = parser.parse_args()

    config_path = args.config if args.config.exists() else Path(__file__).resolve().parent / "config.yaml"
    config = load_yaml(config_path)

    data_dir = Path(args.data_dir or config.get("meddra_data_dir", "./dict/Meddra")).resolve()
    indexes_dir = Path(args.indexes_dir or config.get("indexes_dir", "./indexes")).resolve()
    embedding_conf = config.get("embedding", {})
    vector_store_conf = config.get("vector_store", {})
    vector_backend = vector_store_conf.get("backend", "faiss").lower()

    if not data_dir.exists():
        raise SystemExit(f"MedDRA data directory not found: {data_dir}")

    versions = args.versions or discover_versions(data_dir)
    if not versions:
        raise SystemExit(f"No MedDRA versions found under {data_dir}")

    if vector_backend == "faiss":
        builder = IndexBuilder(
            model_name=embedding_conf.get("model_name", "BAAI/bge-m3"),
            batch_size=int(embedding_conf.get("batch_size", 64)),
            normalize=bool(embedding_conf.get("normalize", True)),
        )
    elif vector_backend == "chroma":
        from modules.vectorstore_chroma import ChromaIndexBuilder  # local import to avoid optional dependency cost

        chroma_conf = vector_store_conf.get("chroma", {})
        builder = ChromaIndexBuilder(
            model_name=embedding_conf.get("model_name", "BAAI/bge-small-en"),
            device=chroma_conf.get("device", embedding_conf.get("device", "auto")),
            collection_prefix=chroma_conf.get("collection_prefix", "meddra"),
            encode_batch_size=int(embedding_conf.get("batch_size", 64)),
            add_batch_size=int(chroma_conf.get("add_batch_size", 2048)),
        )
    else:
        raise SystemExit(f"Unsupported vector_store backend: {vector_backend}")

    for version_name in versions:
        version_dir = data_dir / version_name
        if not version_dir.exists():
            print(f"[WARN] Skipping {version_name}: directory not found at {version_dir}", file=sys.stderr)
            continue

        output_dir = indexes_dir / f"meddra__{version_name}"
        if output_dir.exists() and not args.force:
            print(f"[SKIP] Index already exists for {version_name} at {output_dir}")
            continue

        print(f"[INFO] Parsing MedDRA version: {version_name}")
        parsed = MeddraParser(version_dir).parse()
        print(f"[INFO] Building index into {output_dir}")
        if parsed.documents.empty:
            print(f"[WARN] No documents generated for {version_name}; skipping index build.")
            continue

        metadata = builder.build(
            documents=parsed.documents,
            output_dir=output_dir,
            language=parsed.language,
            version=parsed.version,
        )

        if isinstance(metadata, dict):
            model_name = metadata.get("model_name", "unknown")
            count = metadata.get("documents_count") or metadata.get("terms_count") or parsed.documents.shape[0]
            print(
                f"[DONE] Built index for {version_name} using model {model_name} "
                f"({count} documents)."
            )
        else:
            print(
                f"[DONE] Built index for {version_name} using model {metadata.model_name} "
                f"({metadata.terms_count} documents)."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
